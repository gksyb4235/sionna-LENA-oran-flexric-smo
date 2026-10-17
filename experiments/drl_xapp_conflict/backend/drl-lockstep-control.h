// SPDX-License-Identifier: GPL-2.0-only
#ifndef DRL_LOCKSTEP_CONTROL_H
#define DRL_LOCKSTEP_CONTROL_H

#include "ns3/nr-gnb-net-device.h"
#include "ns3/nr-gnb-rrc.h"
#include "ns3/nr-ue-net-device.h"
#include "ns3/nr-ue-rrc.h"
#include "ns3/node-list.h"
#include "ns3/node.h"
#include "ns3/simulator.h"
#include <pybind11/embed.h>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace ns3
{

/** Isolated, synchronous file transport for research xApps controlling live ns-3. */
class DrlLockstepControl : public std::enable_shared_from_this<DrlLockstepControl>
{
  public:
    /**
     * Install the controller without modifying shared NR code or activating E2.
     * @param cells Canonical labels mapped to the three installed gNBs.
     * @param stopTime Simulation end time in seconds.
     * @param csvPath Live cell KPI CSV, flushed before each observation.
     * @param flush Flush callback for the scenario KPI streams.
     * @return Controller owner, or null if DRL_CONTROL_DIR is absent.
     */
    static std::shared_ptr<DrlLockstepControl>
    Install(const std::map<std::string, Ptr<NrGnbNetDevice>>& cells,
            double stopTime,
            const std::string& csvPath,
            std::function<void()> flush)
    {
        const char* directory = std::getenv("DRL_CONTROL_DIR");
        if (!directory || !*directory)
        {
            return {};
        }
        Require(!std::getenv("AGORA_CONTROL_DIR") && !std::getenv("AGORA_RUN_ID"),
                "AGORA environment must be disabled for DRL lockstep");
        auto self = std::shared_ptr<DrlLockstepControl>(new DrlLockstepControl);
        self->m_directory = std::filesystem::absolute(directory);
        self->m_csvPath = std::filesystem::absolute(csvPath).string();
        self->m_cells = cells;
        self->m_stopTime = stopTime;
        self->m_flush = std::move(flush);
        self->m_interval = EnvironmentNumber("DRL_INTERVAL_S", 30.0);
        self->m_timeout = EnvironmentNumber("DRL_TIMEOUT_S", 300.0);
        const double start = EnvironmentNumber("DRL_START_S", 180.0);
        Require(start > 0 && start < stopTime && self->m_interval > 0 && self->m_timeout > 0,
                "Require 0 < start < stop and positive interval/timeout");
        Require(cells.size() == 3 && cells.count("5g") && cells.count("4g1") && cells.count("4g2"),
                "Exactly the 5g, 4g1 and 4g2 cells are required");
        for (const auto& [label, dev] : cells)
        {
            Require(dev && !dev->GetE2Termination(), "E2 must be disabled on every DRL cell");
        }
        self->m_ues = Ues();
        Require(!self->m_ues.empty(), "No UE RRC objects found");
        std::filesystem::create_directories(self->m_directory);
        for (const auto& name : {"observation.json", "action.json", "receipts.jsonl"})
        {
            Require(!std::filesystem::exists(self->m_directory / name),
                    "Control directory must be fresh: " + self->m_directory.string());
        }
        Simulator::Schedule(Seconds(start), [self]() { self->Tick(false); });
        // Flush the last .5-second KPI row before the scenario stop event.
        Simulator::Schedule(Seconds(stopTime) - NanoSeconds(1), [self]() { self->Tick(true); });
        return self;
    }

  private:
    /**
     * Fail closed instead of silently applying or training on invalid actions.
     * @param valid Required condition.
     * @param message Failure description.
     */
    static void Require(bool valid, const std::string& message)
    {
        if (!valid)
        {
            throw std::runtime_error("DRL lockstep: " + message);
        }
    }

    /**
     * Read a finite environment value with no trailing input.
     * @param name Environment variable name.
     * @param fallback Value used when absent.
     * @return Parsed value.
     */
    static double EnvironmentNumber(const char* name, double fallback)
    {
        const char* text = std::getenv(name);
        if (!text)
        {
            return fallback;
        }
        std::size_t used = 0;
        const double value = std::stod(text, &used);
        Require(used == std::string(text).size() && std::isfinite(value), name);
        return value;
    }

    /** @return All pooled UE RRC instances, including currently idle slots. */
    static std::vector<Ptr<NrUeRrc>> Ues()
    {
        std::vector<Ptr<NrUeRrc>> result;
        for (auto node = NodeList::Begin(); node != NodeList::End(); ++node)
        {
            for (uint32_t i = 0; i < (*node)->GetNDevices(); ++i)
            {
                if (auto ue = DynamicCast<NrUeNetDevice>((*node)->GetDevice(i)))
                {
                    result.push_back(ue->GetRrc());
                }
            }
        }
        return result;
    }

    /** @return Actual CIO values on all UEs and common gNB A3 configuration. */
    std::map<std::string, double> Read() const
    {
        std::map<std::string, double> state;
        bool first = true;
        for (const auto& [label, dev] : m_cells)
        {
            const double cio = m_ues.front()->GetCellIndividualOffset(dev->GetCellId());
            Require(std::isfinite(cio), "Nonfinite CIO readback");
            for (auto ue : m_ues)
            {
                Require(std::abs(ue->GetCellIndividualOffset(dev->GetCellId()) - cio) < 1e-9,
                        "CIO differs between pooled UEs");
            }
            state["cio_" + label] = cio;
            const auto config = dev->GetRrc()->GetA3HandoverConfig();
            Require(std::isfinite(config.first) && std::isfinite(config.second), "No live A3 config");
            if (first)
            {
                state["ttt_ms"] = config.first;
                state["hys_db"] = config.second;
                first = false;
            }
            Require(std::abs(state.at("ttt_ms") - config.first) < 1e-9 &&
                        std::abs(state.at("hys_db") - config.second) < 1e-9,
                    "Global TTT/HYS contract requires equal gNB settings");
        }
        return state;
    }

    /**
     * Convert a finite numerical map to a JSON-compatible Python dictionary.
     * @param values Numerical fields.
     * @return Python dictionary, owned only within the current callback.
     */
    static pybind11::dict Dict(const std::map<std::string, double>& values)
    {
        pybind11::dict result;
        for (const auto& [key, value] : values)
        {
            result[pybind11::str(key)] = value;
        }
        return result;
    }

    /**
     * Publish complete JSON by renaming a sibling temporary file.
     * @param filename Output basename inside this run's control directory.
     * @param value JSON-compatible data.
     */
    void AtomicJson(const std::string& filename, const pybind11::dict& value) const
    {
        const std::string json = pybind11::module_::import("json")
                                     .attr("dumps")(value, pybind11::arg("allow_nan") = false)
                                     .cast<std::string>();
        const auto path = m_directory / filename;
        const auto temporary = m_directory / (filename + ".tmp");
        std::ofstream out(temporary, std::ios::trunc);
        out << json << '\n';
        out.close();
        Require(out.good(), "Cannot flush " + temporary.string());
        std::filesystem::rename(temporary, path);
    }

    /**
     * Append an application/readback record.
     * @param status Applied, terminal or timeout status.
     * @param before State before application.
     * @param updates Validated requested values.
     * @param after State read back after application.
     */
    void Receipt(const std::string& status,
                 const std::map<std::string, double>& before,
                 const std::map<std::string, double>& updates,
                 const std::map<std::string, double>& after) const
    {
        pybind11::dict event;
        event["seq"] = m_seq;
        event["time_s"] = Simulator::Now().GetSeconds();
        event["status"] = status;
        event["before"] = Dict(before);
        event["updates"] = Dict(updates);
        event["after"] = Dict(after);
        pybind11::dict changed;
        bool anyChanged = false;
        for (const auto& [key, value] : after)
        {
            const bool parameterChanged = std::abs(value - before.at(key)) > 1e-9;
            changed[pybind11::str(key)] = parameterChanged;
            anyChanged = anyChanged || parameterChanged;
        }
        event["parameter_changed"] = changed;
        event["any_parameter_changed"] = anyChanged;
        event["ue_count"] = m_ues.size();
        event["verification_scope"] = "all_UE_CIO_and_gNB_A3_config_RRC_reconfiguration_scheduled";
        std::ofstream out(m_directory / "receipts.jsonl", std::ios::app);
        out << pybind11::module_::import("json").attr("dumps")(event).cast<std::string>() << '\n';
        out.close();
        Require(out.good(), "Cannot flush application receipt");
    }

    /**
     * Check the research action space before making any changes.
     * @param key Parameter identifier.
     * @param value Requested target value.
     */
    static void Validate(const std::string& key, double value)
    {
        Require(std::isfinite(value), "Nonfinite action");
        if (key == "cio_5g" || key == "cio_4g1" || key == "cio_4g2")
        {
            Require(value >= -6 && value <= 6, "CIO must be in [-6,6] dB");
            return;
        }
        if (key == "ttt_ms")
        {
            Require(std::set<double>{40, 64, 128, 256, 512, 1024}.count(value), "Invalid TTT action");
            return;
        }
        Require(key == "hys_db" && std::set<double>{0, 1, 2, 3, 4, 6}.count(value),
                "Invalid action parameter or HYS value");
    }

    /**
     * Receive this sequence's complete action with bounded wall-clock waiting.
     * @param before Actual state used for timeout receipt.
     * @return Validated updates, possibly empty for a hold action.
     */
    std::map<std::string, double> WaitAction(const std::map<std::string, double>& before) const
    {
        const auto started = std::chrono::steady_clock::now();
        const auto path = m_directory / "action.json";
        while (std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count() <
               m_timeout)
        {
            if (std::filesystem::exists(path))
            {
                Require(std::filesystem::file_size(path) <= 16384, "Action JSON is too large");
                std::ifstream input(path);
                const std::string payload((std::istreambuf_iterator<char>(input)), {});
                auto decoded = pybind11::module_::import("json").attr("loads")(payload);
                Require(pybind11::isinstance<pybind11::dict>(decoded), "Action must be an object");
                auto data = decoded.cast<pybind11::dict>();
                Require(data.contains("seq") && data.contains("updates"), "Missing action fields");
                Require(pybind11::isinstance<pybind11::int_>(data["seq"]) &&
                            !pybind11::isinstance<pybind11::bool_>(data["seq"]), "Sequence must be integer");
                const auto seq = data["seq"].cast<int64_t>();
                Require(seq <= static_cast<int64_t>(m_seq), "Future action sequence");
                if (seq == static_cast<int64_t>(m_seq))
                {
                    Require(pybind11::isinstance<pybind11::dict>(data["updates"]),
                            "Updates must be an object");
                    auto updates = data["updates"].cast<pybind11::dict>();
                    std::map<std::string, double> result;
                    for (auto item : updates)
                    {
                        Require(pybind11::isinstance<pybind11::str>(item.first), "Invalid parameter key");
                        Require((pybind11::isinstance<pybind11::float_>(item.second) ||
                                 pybind11::isinstance<pybind11::int_>(item.second)) &&
                                    !pybind11::isinstance<pybind11::bool_>(item.second),
                                "Parameter value must be numeric");
                        const auto key = item.first.cast<std::string>();
                        const double value = item.second.cast<double>();
                        Validate(key, value);
                        result[key] = value;
                    }
                    return result;
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        Receipt("timeout", before, {}, before);
        throw std::runtime_error("DRL lockstep: controller action timeout");
    }

    /**
     * Publish past-only observations and apply validated updates in one simulator callback.
     * @param terminal True for final observation; no action is then awaited.
     */
    void Tick(bool terminal)
    {
        m_flush();
        const auto before = Read();
        pybind11::dict observation;
        observation["schema"] = "drl-ns3-lockstep-v1";
        observation["seq"] = m_seq;
        observation["time_s"] = terminal ? m_stopTime : Simulator::Now().GetSeconds();
        observation["terminal"] = terminal;
        observation["cell_kpi_path"] = m_csvPath;
        observation["parameters"] = Dict(before);
        observation["ue_count"] = m_ues.size();
        AtomicJson("observation.json", observation);
        if (terminal)
        {
            Receipt("terminal", before, {}, before);
            return;
        }
        const auto updates = WaitAction(before);
        auto expected = before;
        for (const auto& [key, value] : updates)
        {
            expected[key] = value;
        }
        for (const auto& [label, dev] : m_cells)
        {
            const std::string key = "cio_" + label;
            if (std::abs(expected.at(key) - before.at(key)) > 1e-9)
            {
                for (auto ue : m_ues)
                {
                    ue->SetCellIndividualOffset(dev->GetCellId(), expected.at(key));
                }
            }
            if (expected.at("ttt_ms") != before.at("ttt_ms") ||
                expected.at("hys_db") != before.at("hys_db"))
            {
                // This updates A3 and schedules RRC reconfiguration for every connected UE.
                dev->GetRrc()->UpdateA3HandoverConfig(expected.at("ttt_ms"), expected.at("hys_db"));
            }
        }
        const auto after = Read();
        for (const auto& [key, value] : expected)
        {
            Require(std::abs(after.at(key) - value) < 1e-9, "Application readback mismatch: " + key);
        }
        Receipt(updates.empty() ? "held" : "applied", before, updates, after);
        ++m_seq;
        if (Simulator::Now() + Seconds(m_interval) < Seconds(m_stopTime) - NanoSeconds(1))
        {
            auto self = shared_from_this();
            Simulator::Schedule(Seconds(m_interval), [self]() { self->Tick(false); });
        }
    }

    std::filesystem::path m_directory; ///< Per-run control directory.
    std::string m_csvPath; ///< Absolute scenario KPI CSV path.
    std::map<std::string, Ptr<NrGnbNetDevice>> m_cells; ///< Live gNB objects.
    std::vector<Ptr<NrUeRrc>> m_ues; ///< Preallocated UE RRC objects.
    std::function<void()> m_flush; ///< Scenario stream flushing callback.
    double m_stopTime{0}; ///< Simulation stop time in seconds.
    double m_interval{30}; ///< Decision interval in simulation seconds.
    double m_timeout{300}; ///< Maximum controller wait in wall seconds.
    uint64_t m_seq{0}; ///< Monotonically increasing observation sequence.
};

} // namespace ns3
#endif // DRL_LOCKSTEP_CONTROL_H

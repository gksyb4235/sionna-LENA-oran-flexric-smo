// SPDX-License-Identifier: GPL-2.0-only
#ifndef AGORA_CONFLICT_SCHEDULE_CONTROL_H
#define AGORA_CONFLICT_SCHEDULE_CONTROL_H

#include "ns3/double.h"
#include "ns3/mobility-model.h"
#include "ns3/node-list.h"
#include "ns3/node.h"
#include "ns3/nr-gnb-net-device.h"
#include "ns3/nr-gnb-phy.h"
#include "ns3/nr-gnb-rrc.h"
#include "ns3/nr-spectrum-phy.h"
#include "ns3/nr-ue-net-device.h"
#include "ns3/nr-ue-rrc.h"
#include "ns3/simulator.h"
#include "ns3/sionna-lookup-channel-model.h"
#include "ns3/uniform-planar-array.h"
#include <pybind11/embed.h>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <initializer_list>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ns3
{

/** Predetermined grouped simulator changes; no inference, E2 transport or wall-clock waiting. */
class ConflictScheduleControl
{
  public:
    /** Parameters changed together in one simulator callback. */
    struct Event
    {
        double time; ///< Absolute simulator seconds.
        std::map<std::string, double> updates; ///< Absolute physical target values.
    };

    /**
     * Parse CSV fields time_s,parameter,value; equal timestamps form one event.
     * @param path Schedule path; empty disables scheduled changes.
     * @param stopTime Simulation duration.
     * @return Events with strictly increasing group times and unique keys.
     */
    static std::vector<Event> Load(const std::string& path, double stopTime)
    {
        if (path.empty())
        {
            return {};
        }
        Require(std::isfinite(stopTime) && stopTime > 0, "Invalid simulation duration");
        std::ifstream input(path);
        Require(input.good(), "Cannot open schedule: " + path);
        std::vector<Event> events;
        std::string line;
        bool header = false;
        while (std::getline(input, line))
        {
            line = Trim(line);
            if (line.empty())
            {
                continue;
            }
            if (!header)
            {
                Require(line == "time_s,parameter,value", "Expected CSV header time_s,parameter,value");
                header = true;
                continue;
            }
            std::stringstream row(line);
            std::vector<std::string> fields;
            std::string field;
            while (std::getline(row, field, ','))
            {
                fields.push_back(Trim(field));
            }
            Require(fields.size() == 3 && line.back() != ',', "Invalid CSV field count");
            const double time = Number(fields[0]);
            const double value = Number(fields[2]);
            Validate(fields[1], value);
            Require(time > 0 && time < stopTime, "Event time must satisfy 0 < time < simTime");
            Require(events.empty() || Seconds(time) >= Seconds(events.back().time),
                    "Schedule times must be nondecreasing");
            if (events.empty() || Seconds(time) != Seconds(events.back().time))
            {
                events.push_back({time, {}});
            }
            Require(events.back().time == time,
                    "Distinct timestamps round to the same simulator time");
            Require(events.back().updates.emplace(fields[1], value).second,
                    "Duplicate parameter in one event: " + fields[1]);
        }
        Require(header && !events.empty(), "Schedule must contain at least one event");
        return events;
    }

    /**
     * Prevalidate the entire schedule and exact RET caches, then install callbacks.
     * @param events Parsed grouped changes.
     * @param cells Canonical labels mapped to installed gNBs.
     * @param lookups Lookup channels for each cell bandwidth part.
     * @param cacheRoot Root of exact tilt_Ndeg cache files.
     * @param outputPath Physical application receipt CSV.
     * @return Owner that must live through Simulator::Run, or null for no changes.
     */
    static std::shared_ptr<ConflictScheduleControl>
    Install(const std::vector<Event>& events,
            const std::map<std::string, Ptr<NrGnbNetDevice>>& cells,
            const std::map<std::string, std::vector<Ptr<SionnaLookupChannelModel>>>& lookups,
            const std::string& cacheRoot,
            const std::string& outputPath)
    {
        if (events.empty())
        {
            return {};
        }
        Require(!std::getenv("AGORA_CONTROL_DIR") && !std::getenv("AGORA_RUN_ID") &&
                    !std::getenv("DRL_CONTROL_DIR"), "External controller environment is forbidden");
        auto self = std::shared_ptr<ConflictScheduleControl>(new ConflictScheduleControl);
        self->m_cells = cells;
        self->m_lookups = lookups;
        Require(cells.size() == 3 && cells.count("5g") && cells.count("4g1") && cells.count("4g2"),
                "Exactly three canonical cells are required");
        for (const auto& [label, dev] : cells)
        {
            Require(dev && !dev->GetE2Termination(), "Every cell must have E2 disabled");
        }
        for (auto node = NodeList::Begin(); node != NodeList::End(); ++node)
        {
            for (uint32_t i = 0; i < (*node)->GetNDevices(); ++i)
            {
                if (auto ue = DynamicCast<NrUeNetDevice>((*node)->GetDevice(i)))
                {
                    self->m_ues.push_back(ue->GetRrc());
                }
            }
        }
        Require(!self->m_ues.empty(), "No pooled UE RRC instances found");
        auto predicted = self->ReadAll();
        for (const auto& [key, value] : predicted)
        {
            Validate(key, value);
        }
        std::set<std::string> validatedInitial;
        for (const auto& event : events)
        {
            for (const auto& [key, value] : event.updates)
            {
                Require(std::abs(predicted.at(key) - value) > 1e-7, "No-op event: " + key);
                if (key.rfind("ret_", 0) == 0)
                {
                    const auto label = key.substr(4);
                    const auto dev = cells.at(label);
                    Require(lookups.count(label) && !lookups.at(label).empty(), "Missing RET lookup");
                    for (auto lookup : lookups.at(label))
                    {
                        Require(lookup != nullptr, "Invalid RET lookup");
                        if (!validatedInitial.count(label))
                        {
                            self->ValidateCache(lookup->GetCacheFile(), CellName(label), dev,
                                                self->ReadAll().at(key), lookup->GetFrequency());
                        }
                        const auto band = lookup->GetFrequency() < 2e9 ? "1p8ghz" : "3p5ghz";
                        const auto filename = std::string("sionna_rt_cache_") + band + "_2x2_1x1_lzf.h5";
                        auto target = std::filesystem::path(cacheRoot) /
                                      ("tilt_" + std::to_string(static_cast<int>(value)) + "deg") / filename;
                        self->ValidateCache(target.string(), CellName(label), dev, value,
                                            lookup->GetFrequency());
                        Require(lookup->PrepareRetCache(CellName(label), value, cacheRoot),
                                "Cannot prepare exact RET cache: " + target.string());
                    }
                    validatedInitial.insert(label);
                }
                predicted[key] = value;
            }
        }
        Require(!std::filesystem::exists(outputPath), "Receipt already exists: " + outputPath);
        self->m_output.open(outputPath, std::ios::out);
        Require(self->m_output.good(), "Cannot create receipt: " + outputPath);
        self->m_output << "scheduled_time_s,applied_time_s,parameter,before,after,status\n";
        self->m_output.flush();
        for (const auto& event : events)
        {
            Require(Seconds(event.time) > Simulator::Now(), "Event is already in the past");
            Simulator::Schedule(Seconds(event.time) - Simulator::Now(),
                                [self, event]() { self->Apply(event); });
        }
        return self;
    }

  private:
    /** @param valid Required condition. @param message Failure reason. */
    static void Require(bool valid, const std::string& message)
    {
        if (!valid)
        {
            throw std::runtime_error("conflict schedule: " + message);
        }
    }

    /** @param value CSV text. @return Text without surrounding whitespace. */
    static std::string Trim(std::string value)
    {
        const auto keep = [](unsigned char c) { return !std::isspace(c); };
        value.erase(value.begin(), std::find_if(value.begin(), value.end(), keep));
        value.erase(std::find_if(value.rbegin(), value.rend(), keep).base(), value.end());
        return value;
    }

    /** @param field CSV numeric field. @return Complete finite value. */
    static double Number(const std::string& field)
    {
        std::size_t end = 0;
        const double value = std::stod(field, &end);
        Require(end == field.size() && std::isfinite(value), "Invalid finite number: " + field);
        return value;
    }

    /** @param label Canonical cell label. @return Exact HDF5 group name. */
    static std::string CellName(const std::string& label)
    {
        static const std::map<std::string, std::string> names{
            {"5g", "gNB_5G"}, {"4g1", "gNB_4G_1"}, {"4g2", "gNB_4G_2"}};
        Require(names.count(label), "Unknown cell label: " + label);
        return names.at(label);
    }

    /** @param key Parameter key. @param value Value in the frozen research domain. */
    static void Validate(const std::string& key, double value)
    {
        Require(std::isfinite(value), "Nonfinite parameter");
        if (key == "ttt_ms")
        {
            Require(std::set<double>{128, 256, 340, 512, 720}.count(value), "Unsupported TTT");
            return;
        }
        if (key == "hys_db")
        {
            Require(value >= 0.5 && value <= 5 && std::abs(2 * value - std::round(2 * value)) < 1e-9,
                    "HYS requires [0.5,5] in 0.5 dB steps");
            return;
        }
        Require(key.size() > 4, "Unknown parameter: " + key);
        CellName(key.substr(4));
        const auto family = key.substr(0, 4);
        if (family == "txp_")
        {
            Require(InDomain(value, {30, 33, 37, 40, 43}), "Unsupported TxP");
        }
        else if (family == "ret_")
        {
            Require(InDomain(value, {3, 5, 7, 9, 12}), "Unsupported exact RET cache angle");
        }
        else
        {
            Require(family == "cio_", "Unknown parameter: " + key);
            const auto low = key == "cio_5g" ? 5 : -2;
            const auto high = key == "cio_5g" ? 9 : 2;
            Require(value >= low && value <= high && std::abs(value - std::round(value)) < 1e-9,
                    "CIO outside the frozen integer domain");
        }
    }

    /** @param value Physical readback. @param domain Allowed targets. @return Membership with roundoff tolerance. */
    static bool InDomain(double value, std::initializer_list<double> domain)
    {
        return std::any_of(domain.begin(), domain.end(),
                           [value](double target) { return std::abs(value - target) < 1e-7; });
    }

    /**
     * Validate an exact cache against live site, band and antenna orientation.
     * @param path HDF5 file. @param cell Cache group. @param dev Live gNB.
     * @param tilt Exact angle in degrees. @param frequency Carrier frequency.
     */
    static void ValidateCache(const std::string& path, const std::string& cell,
                              Ptr<NrGnbNetDevice> dev, double tilt, double frequency)
    {
        Require(std::filesystem::is_regular_file(path), "Missing cache: " + path);
        namespace py = pybind11;
        py::object file = py::module_::import("h5py").attr("File")(path, "r");
        try
        {
            Require(file.attr("__contains__")(cell).cast<bool>(), "Cache has no cell " + cell);
            py::object attrs = file[py::str(cell)].attr("attrs");
            Require(std::abs(attrs["tilt_deg"].cast<double>() - tilt) < 1e-6, "Cache tilt mismatch");
            Require(std::abs(attrs["frequency_hz"].cast<double>() - frequency) < 1, "Cache band mismatch");
            const auto position = dev->GetNode()->GetObject<MobilityModel>()->GetPosition();
            py::object stored = attrs["gnb_position"];
            const Vector cached(stored[py::int_(0)].cast<double>(), stored[py::int_(1)].cast<double>(),
                                stored[py::int_(2)].cast<double>());
            Require(CalculateDistance(position, cached) < 0.01, "Cache site mismatch");
            for (uint32_t cc = 0; cc < dev->GetCcMapSize(); ++cc)
            {
                auto antenna = DynamicCast<UniformPlanarArray>(dev->GetPhy(cc)->GetSpectrumPhy()->GetAntenna());
                Require(antenna != nullptr, "RET requires a UniformPlanarArray");
                DoubleValue bearing;
                antenna->GetAttribute("BearingAngle", bearing);
                Require(std::abs(std::remainder(attrs["bearing_deg"].cast<double>() -
                                  bearing.Get() * 180.0 / M_PI, 360.0)) < 1e-6, "Cache bearing mismatch");
            }
        }
        catch (...)
        {
            file.attr("close")();
            throw;
        }
        file.attr("close")();
    }

    /** @return Actual eleven settings; verifies all carriers, pooled UE CIO and common gNB A3. */
    std::map<std::string, double> ReadAll() const
    {
        std::map<std::string, double> result;
        bool first = true;
        for (const auto& [label, dev] : m_cells)
        {
            Require(dev->GetCcMapSize() > 0, "Cell has no component carrier");
            const auto cio = m_ues.front()->GetCellIndividualOffset(dev->GetCellId());
            for (auto ue : m_ues)
            {
                Require(std::abs(ue->GetCellIndividualOffset(dev->GetCellId()) - cio) < 1e-9,
                        "CIO differs between pooled UEs");
            }
            result["cio_" + label] = cio;
            const auto config = dev->GetRrc()->GetA3HandoverConfig();
            if (first)
            {
                result["ttt_ms"] = config.first;
                result["hys_db"] = config.second;
                first = false;
            }
            Require(std::abs(result.at("ttt_ms") - config.first) < 1e-9 &&
                        std::abs(result.at("hys_db") - config.second) < 1e-9,
                    "TTT/HYS must be shared across all gNBs");
            for (uint32_t cc = 0; cc < dev->GetCcMapSize(); ++cc)
            {
                const auto power = dev->GetPhy(cc)->GetTxPower();
                auto antenna = DynamicCast<UniformPlanarArray>(dev->GetPhy(cc)->GetSpectrumPhy()->GetAntenna());
                Require(antenna != nullptr, "RET requires a UniformPlanarArray");
                DoubleValue tilt;
                antenna->GetAttribute("DowntiltAngle", tilt);
                const double angle = tilt.Get() * 180.0 / M_PI;
                if (cc == 0)
                {
                    result["txp_" + label] = power;
                    result["ret_" + label] = angle;
                }
                Require(std::abs(power - result.at("txp_" + label)) < 1e-6 &&
                            std::abs(angle - result.at("ret_" + label)) < 1e-6,
                        "Component carrier settings disagree");
            }
        }
        for (const auto& [key, value] : result)
        {
            Require(std::isfinite(value), "Nonfinite live parameter: " + key);
        }
        return result;
    }

    /**
     * Apply a whole event before another simulator event can run, then verify all eleven fields.
     * Atomicity refers to simulator callbacks, not rollback or UE RRC acknowledgement.
     * @param event Prevalidated simultaneous changes.
     */
    void Apply(const Event& event)
    {
        const auto before = ReadAll();
        auto expected = before;
        for (const auto& [key, value] : event.updates)
        {
            expected[key] = value;
            if (key.rfind("ret_", 0) == 0)
            {
                const auto label = key.substr(4);
                for (auto lookup : m_lookups.at(label))
                {
                    lookup->ActivateRetCache(CellName(label), value);
                }
            }
        }
        for (const auto& [label, dev] : m_cells)
        {
            for (uint32_t cc = 0; cc < dev->GetCcMapSize(); ++cc)
            {
                if (event.updates.count("txp_" + label))
                {
                    dev->GetPhy(cc)->SetTxPower(expected.at("txp_" + label));
                }
                if (event.updates.count("ret_" + label))
                {
                    DynamicCast<UniformPlanarArray>(dev->GetPhy(cc)->GetSpectrumPhy()->GetAntenna())
                        ->SetAttribute("DowntiltAngle", DoubleValue(expected.at("ret_" + label) * M_PI / 180));
                }
            }
            if (event.updates.count("cio_" + label))
            {
                for (auto ue : m_ues)
                {
                    ue->SetCellIndividualOffset(dev->GetCellId(), expected.at("cio_" + label));
                }
            }
            if (event.updates.count("ttt_ms") || event.updates.count("hys_db"))
            {
                dev->GetRrc()->UpdateA3HandoverConfig(expected.at("ttt_ms"), expected.at("hys_db"));
            }
        }
        const auto after = ReadAll();
        bool valid = Simulator::Now() == Seconds(event.time);
        for (const auto& [key, value] : expected)
        {
            valid = valid && std::abs(after.at(key) - value) < 1e-6;
        }
        for (const auto& [key, value] : event.updates)
        {
            m_output << std::setprecision(17) << event.time << ',' << Simulator::Now().GetSeconds()
                     << ',' << key << ',' << before.at(key) << ',' << after.at(key) << ','
                     << (valid ? "applied" : "readback_failed") << '\n';
        }
        m_output.flush();
        Require(m_output.good(), "Cannot flush application receipt");
        Require(valid, "Group readback or exact application time failed");
    }

    std::map<std::string, Ptr<NrGnbNetDevice>> m_cells; ///< Installed gNBs.
    std::map<std::string, std::vector<Ptr<SionnaLookupChannelModel>>> m_lookups; ///< Exact cell channels.
    std::vector<Ptr<NrUeRrc>> m_ues; ///< All pooled UEs, including currently idle slots.
    std::ofstream m_output; ///< Flushed application receipts; not end-to-end RRC acknowledgements.
};

} // namespace ns3
#endif // AGORA_CONFLICT_SCHEDULE_CONTROL_H

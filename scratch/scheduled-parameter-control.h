// SPDX-License-Identifier: GPL-2.0-only
#ifndef SCHEDULED_PARAMETER_CONTROL_H
#define SCHEDULED_PARAMETER_CONTROL_H

#include "ns3/double.h"
#include "ns3/mobility-model.h"
#include "ns3/node.h"
#include "ns3/nr-gnb-net-device.h"
#include "ns3/nr-gnb-phy.h"
#include "ns3/nr-spectrum-phy.h"
#include "ns3/simulator.h"
#include "ns3/sionna-lookup-channel-model.h"
#include "ns3/sionna-rt-spectrum-propagation-loss-model.h"
#include "ns3/spectrum-channel.h"
#include "ns3/uniform-planar-array.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ns3
{

/** Permanent, predetermined simulator events, independent of external controllers. */
class ScheduledParameterControl
{
  public:
    /** One CSV event. */
    struct Event
    {
        double time;           ///< Absolute simulation seconds.
        std::string parameter; ///< Canonical txp_CELL or ret_CELL name.
        double value;          ///< dBm for TxP, degrees for RET.
    };

    /**
     * Parse and validate the schedule before building the simulation.
     * @param path CSV path, or empty to disable scheduling.
     * @param stopTime Simulation stop time in seconds.
     * @return Validated events in strictly increasing time order.
     */
    static std::vector<Event> Load(const std::string& path, double stopTime)
    {
        if (path.empty())
        {
            return {};
        }
        Require(std::isfinite(stopTime) && stopTime > 0, "Invalid simulation stop time");
        std::ifstream input(path);
        Require(input.good(), "Cannot open parameter schedule: " + path);
        std::vector<Event> events;
        std::string line;
        bool header = false;
        unsigned lineNumber = 0;
        while (std::getline(input, line))
        {
            ++lineNumber;
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
            std::vector<std::string> fields;
            std::stringstream row(line);
            std::string field;
            while (std::getline(row, field, ','))
            {
                fields.push_back(Trim(field));
            }
            Require(fields.size() == 3 && line.back() != ',',
                    "Invalid schedule CSV row " + std::to_string(lineNumber));
            Event event{Number(fields[0]), fields[1], Number(fields[2])};
            Require(event.time > 0 && event.time < stopTime,
                    "Schedule time must be greater than zero and less than simTime");
            Require(events.empty() || Seconds(event.time) > Seconds(events.back().time),
                    "Schedule times must be strictly increasing at simulator resolution");
            const bool txp = event.parameter.rfind("txp_", 0) == 0;
            const bool ret = event.parameter.rfind("ret_", 0) == 0;
            Require(txp || ret, "Unsupported parameter: " + event.parameter);
            CellName(event.parameter.substr(4));
            Require(txp ? event.value >= 0 && event.value <= 60
                        : event.value >= 0 && event.value <= 90 &&
                              std::abs(event.value - std::round(event.value)) < 1e-9,
                    "TxP requires 0..60 dBm; RET requires an integer 0..90 degree exact cache");
            events.push_back(event);
        }
        Require(header && !events.empty(), "Parameter schedule contains no events");
        return events;
    }

    /**
     * Validate physical state and preload every RET cache, then schedule events.
     * @param events Parsed events.
     * @param cells Canonical cell labels mapped to installed gNB devices.
     * @param lookups Per-cell lookup objects from the configured bandwidth parts.
     * @param cacheRoot Root containing tilt_Ndeg directories.
     * @param outputPath CSV application receipt path.
     * @return Owner that must remain alive through Simulator::Run().
     */
    static std::shared_ptr<ScheduledParameterControl>
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
        Require(!std::getenv("AGORA_CONTROL_DIR") && !std::getenv("AGORA_RUN_ID"),
                "Predetermined scheduling cannot run with an external controller environment");
        auto controller = std::shared_ptr<ScheduledParameterControl>(new ScheduledParameterControl);
        controller->m_cells = cells;
        controller->m_lookups = lookups;
        std::map<std::string, double> predicted;
        for (const auto& event : events)
        {
            const std::string label = event.parameter.substr(4);
            const auto found = cells.find(label);
            Require(found != cells.end() && found->second, "Unknown live cell: " + label);
            Require(!found->second->GetE2Termination(), "Predetermined scheduling requires E2 disabled");
            const double initial = controller->Read(event.parameter);
            const double before = predicted.count(event.parameter) ? predicted.at(event.parameter) : initial;
            Require(std::abs(before - event.value) > 1e-7,
                    "No-op schedule event: " + event.parameter);
            predicted[event.parameter] = event.value;
            if (event.parameter.rfind("ret_", 0) == 0)
            {
                const auto dev = found->second;
                Require(lookups.count(label) && !lookups.at(label).empty(),
                        "No Sionna lookup is connected to the RET cell");
                for (auto lookup : lookups.at(label))
                {
                    Require(lookup != nullptr, "Invalid cell lookup");
                    ValidateCache(lookup->GetCacheFile(), CellName(label), dev, initial,
                                  lookup->GetFrequency());
                    const std::string band = lookup->GetFrequency() < 2e9 ? "1p8ghz" : "3p5ghz";
                    const std::string filename = "sionna_rt_cache_" + band + "_2x2_1x1_lzf.h5";
                    std::filesystem::path target =
                        std::filesystem::path(cacheRoot) /
                        ("tilt_" + std::to_string(static_cast<int>(event.value)) + "deg") / filename;
                    if (event.value == 15)
                    {
                        target = std::filesystem::path(cacheRoot).parent_path() / filename;
                    }
                    // Match the existing 15-degree baseline location; never substitute a tilt.
                    ValidateCache(target.string(), CellName(label), dev, event.value,
                                  lookup->GetFrequency());
                    Require(lookup->PrepareRetCache(CellName(label), event.value, cacheRoot),
                            "Cannot prepare exact RET cache: " + target.string());
                }
            }
        }
        controller->m_output.open(outputPath, std::ios::out | std::ios::trunc);
        Require(controller->m_output.good(), "Cannot write parameter receipt: " + outputPath);
        controller->m_output << "scheduled_time_s,applied_time_s,parameter,before,after,status\n";
        controller->m_output.flush();
        for (const auto& event : events)
        {
            Require(Seconds(event.time) > Simulator::Now(), "Schedule event is already in the past");
            Simulator::Schedule(Seconds(event.time) - Simulator::Now(),
                                [controller, event]() { controller->Apply(event); });
        }
        return controller;
    }

  private:
    /** Reject invalid experiment inputs with a useful message. */
    static void Require(bool valid, const std::string& message)
    {
        if (!valid)
        {
            throw std::runtime_error("parameter schedule: " + message);
        }
    }

    /** Remove surrounding CSV whitespace. */
    static std::string Trim(std::string value)
    {
        auto nonSpace = [](unsigned char c) { return !std::isspace(c); };
        value.erase(value.begin(), std::find_if(value.begin(), value.end(), nonSpace));
        value.erase(std::find_if(value.rbegin(), value.rend(), nonSpace).base(), value.end());
        return value;
    }

    /** Parse a complete finite numeric CSV field. */
    static double Number(const std::string& field)
    {
        std::size_t end = 0;
        const double value = std::stod(field, &end);
        Require(end == field.size() && std::isfinite(value), "Invalid finite number: " + field);
        return value;
    }

    /** Map a canonical cell to its cache group, rejecting unknown labels. */
    static std::string CellName(const std::string& label)
    {
        static const std::map<std::string, std::string> names{
            {"5g", "gNB_5G"}, {"4g1", "gNB_4G_1"}, {"4g2", "gNB_4G_2"}};
        Require(names.count(label), "Unknown cell label: " + label);
        return names.at(label);
    }

    /** Validate that a cache represents the exact live site, band and antenna orientation. */
    static void ValidateCache(const std::string& path,
                              const std::string& cell,
                              Ptr<NrGnbNetDevice> dev,
                              double tilt,
                              double frequency)
    {
        Require(std::filesystem::is_regular_file(path), "Missing exact RET cache: " + path);
        py::object file = py::module_::import("h5py").attr("File")(path, "r");
        try
        {
            Require(file.attr("__contains__")(cell).cast<bool>(), "Cache has no cell " + cell);
            py::object attrs = file[py::str(cell)].attr("attrs");
            Require(std::abs(attrs["tilt_deg"].cast<double>() - tilt) < 1e-6,
                    "Cache tilt mismatch: " + path);
            Require(std::abs(attrs["frequency_hz"].cast<double>() - frequency) < 1.0,
                    "Cache frequency mismatch: " + path);
            const auto position = dev->GetNode()->GetObject<MobilityModel>()->GetPosition();
            py::object stored = attrs["gnb_position"];
            const Vector cached(stored[py::int_(0)].cast<double>(),
                                stored[py::int_(1)].cast<double>(),
                                stored[py::int_(2)].cast<double>());
            Require(CalculateDistance(position, cached) < 0.01, "Cache site mismatch: " + path);
            for (uint32_t cc = 0; cc < dev->GetCcMapSize(); ++cc)
            {
                auto antenna = DynamicCast<UniformPlanarArray>(
                    dev->GetPhy(cc)->GetSpectrumPhy()->GetAntenna());
                Require(antenna != nullptr, "RET requires a UniformPlanarArray");
                DoubleValue bearing;
                antenna->GetAttribute("BearingAngle", bearing);
                Require(std::abs(std::remainder(attrs["bearing_deg"].cast<double>() -
                                                   bearing.Get() * 180.0 / M_PI, 360.0)) < 1e-6,
                        "Cache bearing mismatch: " + path);
            }
        }
        catch (...)
        {
            file.attr("close")();
            throw;
        }
        file.attr("close")();
    }

    /** Read all carrier values and reject inconsistent or non-finite physical state. */
    double Read(const std::string& parameter) const
    {
        const auto dev = m_cells.at(parameter.substr(4));
        Require(dev->GetCcMapSize() > 0, "Cell has no component carriers");
        double first = 0;
        for (uint32_t cc = 0; cc < dev->GetCcMapSize(); ++cc)
        {
            double value = dev->GetPhy(cc)->GetTxPower();
            if (parameter.rfind("ret_", 0) == 0)
            {
                auto antenna = DynamicCast<UniformPlanarArray>(
                    dev->GetPhy(cc)->GetSpectrumPhy()->GetAntenna());
                Require(antenna != nullptr, "RET requires a UniformPlanarArray");
                DoubleValue tilt;
                antenna->GetAttribute("DowntiltAngle", tilt);
                value = tilt.Get() * 180.0 / M_PI;
            }
            Require(std::isfinite(value), "Non-finite live parameter");
            if (cc == 0)
            {
                first = value;
            }
            Require(std::abs(value - first) < 1e-6, "Component-carrier parameters disagree");
        }
        return first;
    }

    /** Apply one permanent change, verify physical readback, and flush its receipt. */
    void Apply(const Event& event)
    {
        const double before = Read(event.parameter);
        const auto label = event.parameter.substr(4);
        const auto dev = m_cells.at(label);
        if (event.parameter.rfind("ret_", 0) == 0)
        {
            for (auto lookup : m_lookups.at(label))
            {
                lookup->ActivateRetCache(CellName(label), event.value);
            }
        }
        for (uint32_t cc = 0; cc < dev->GetCcMapSize(); ++cc)
        {
            if (event.parameter.rfind("txp_", 0) == 0)
            {
                dev->GetPhy(cc)->SetTxPower(event.value);
            }
            else
            {
                DynamicCast<UniformPlanarArray>(dev->GetPhy(cc)->GetSpectrumPhy()->GetAntenna())
                    ->SetAttribute("DowntiltAngle", DoubleValue(event.value * M_PI / 180.0));
            }
        }
        const double after = Read(event.parameter);
        const bool valid = std::abs(after - event.value) < 1e-6 &&
                           Simulator::Now() == Seconds(event.time);
        m_output << std::setprecision(17) << event.time << ',' << Simulator::Now().GetSeconds()
                 << ',' << event.parameter << ',' << before << ',' << after << ','
                 << (valid ? "applied" : "readback_failed") << '\n';
        m_output.flush();
        Require(m_output.good(), "Failed to flush parameter application receipt");
        Require(valid, "Physical parameter readback or application time differs from schedule");
    }

    std::map<std::string, Ptr<NrGnbNetDevice>> m_cells; ///< Live gNBs.
    std::map<std::string, std::vector<Ptr<SionnaLookupChannelModel>>> m_lookups; ///< Cell channels.
    std::ofstream m_output; ///< Flushed physical application receipts.
};

} // namespace ns3

#endif // SCHEDULED_PARAMETER_CONTROL_H

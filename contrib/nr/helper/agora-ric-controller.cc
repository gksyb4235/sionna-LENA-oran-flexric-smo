// SPDX-License-Identifier: GPL-2.0-only
#include "agora-ric-controller.h"
#include "ns3/nr-gnb-net-device.h"
#include "ns3/nr-gnb-phy.h"
#include "ns3/nr-gnb-rrc.h"
#include "ns3/nr-rrc-protocol-ideal.h"
#include "ns3/nr-spectrum-phy.h"
#include "ns3/nr-ue-net-device.h"
#include "ns3/nr-ue-rrc.h"
#include "ns3/node-list.h"
#include "ns3/node.h"
#include "ns3/simulator.h"
#include "ns3/uniform-planar-array.h"
#include "ns3/sionna-lookup-channel-model.h"
#include "ns3/double.h"
#include "ns3/log.h"
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <set>
#include <sstream>
#include <thread>
#include <unistd.h>

namespace ns3
{
NS_LOG_COMPONENT_DEFINE("AgoraRicController");
namespace
{
struct Change
{
    uint16_t cell;
    int kind;
    double value;
    double expected;
};
struct Command
{
    std::string id;
    double start;
    double end;
    std::vector<Change> checks;
    std::vector<Change> changes;
};
std::map<std::string, Ptr<NrGnbNetDevice>> g_cells;
std::map<uint16_t, Ptr<NrGnbNetDevice>> g_byId;
std::vector<Ptr<SionnaLookupChannelModel>> g_lookups;
std::map<std::string, std::string> g_seen;
std::string g_dir, g_run, g_replay, g_retRoot;
double g_stop = 0;
std::string g_lease;
const char* kNames[] = {"", "txp", "ret", "cio", "ttt_ms", "hys_db"};

bool SafeId(const std::string& id)
{
    return id.size() == 32 && id.find_first_not_of("0123456789abcdef") == std::string::npos;
}
std::vector<Ptr<NrUeRrc>> Ues()
{
    std::vector<Ptr<NrUeRrc>> result;
    for (auto n = NodeList::Begin(); n != NodeList::End(); ++n)
        for (uint32_t i=0; i<(*n)->GetNDevices(); ++i)
            if (auto ue = DynamicCast<NrUeNetDevice>((*n)->GetDevice(i))) result.push_back(ue->GetRrc());
    return result;
}
double Read(uint16_t cell, int kind)
{
    auto dev = g_byId.at(cell);
    if (kind == 1) return dev->GetPhy(0)->GetTxPower();
    if (kind == 2)
    {
        auto antenna = DynamicCast<UniformPlanarArray>(dev->GetPhy(0)->GetSpectrumPhy()->GetAntenna());
        DoubleValue angle;
        antenna->GetAttribute("DowntiltAngle", angle);
        return angle.Get()*180.0/M_PI;
    }
    if (kind == 3)
    {
        auto ues=Ues();
        return ues.empty() ? 0.0 : ues.front()->GetCellIndividualOffset(cell);
    }
    auto config = dev->GetRrc()->GetA3HandoverConfig();
    return kind == 4 ? config.first : config.second;
}
bool Equal(double a, double b) { return std::abs(a-b) < 0.001; }
bool Domain(int kind, double value)
{
    if (!std::isfinite(value)) return false;
    if (kind == 1) return value >= 0 && value <= 60;
    if (kind == 2) return value >= 0 && value <= 20;
    if (kind == 3) return value >= -15 && value <= 15;
    if (kind == 5) return value >= 0 && value <= 15 && Equal(value*2, std::round(value*2));
    // Ideal RRC supports arbitrary integer milliseconds, including the training values.
    // Real RRC deployments are restricted separately to encodable 3GPP values.
    return kind == 4 && value >= 0 && value <= 5120 && Equal(value, std::round(value));
}
void Snapshot(bool finished=false)
{
    if (g_dir.empty()) return;
    std::ofstream out(g_dir+"/status.json.tmp");
    out << std::setprecision(15) << "{\"schema\":\"agora-ns3-live-v1\",\"run_id\":\"" << g_run
        << "\",\"pid\":" << getpid() << ",\"sim_time_s\":" << Simulator::Now().GetSeconds()
        << ",\"stop_time_s\":" << g_stop << ",\"finished\":" << (finished?"true":"false")
        << ",\"wall_time_s\":" << std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count()
        << ",\"replay_run_id\":\"" << g_replay << "\",\"lease\":\"" << g_lease << "\",\"cells\":[";
    bool first=true;
    for (auto& [label,dev] : g_cells)
    {
        if (!first) out << ',';
        first=false;
        out << "{\"label\":\"" << label << "\",\"cell_id\":" << dev->GetCellId() << ",\"e2_enabled\":"
            << (dev->GetE2Termination()?"true":"false") << ",\"ideal_rrc\":"
            << (dev->GetRrc()->GetObject<NrGnbRrcProtocolIdeal>()?"true":"false") << ",\"parameters\":{";
        for (int kind=1;kind<=5;++kind)
        {
            if(kind>1) out << ',';
            double value=Read(dev->GetCellId(),kind);
            out << '\"' << kNames[kind] << "\":";
            if(std::isfinite(value)) out << value; else out << "null";
        }
        out << "}}";
    }
    out << "],\"receipt_transport\":\"local_ns3_readback_after_E2\"}";
    out.close();
    std::filesystem::rename(g_dir+"/status.json.tmp",g_dir+"/status.json");
}
void Event(const std::string& id, const std::string& state, const std::string& reason="")
{
    if (g_dir.empty()) return;
    std::ofstream out(g_dir+"/receipts.jsonl",std::ios::app);
    out << std::setprecision(15) << "{\"run_id\":\"" << g_run << "\",\"command_id\":\"" << id
        << "\",\"status\":\"" << state << "\",\"sim_time_s\":" << Simulator::Now().GetSeconds()
        << ",\"reason\":\"" << reason << "\",\"readback\":[";
    bool first=true;
    for(auto& [cell,dev]:g_byId) for(int kind=1;kind<=5;++kind)
    {
        if(!first)out<<',';
        first=false;
        double value=Read(cell,kind);
        out << "{\"cell\":" << cell << ",\"kind\":" << kind << ",\"value\":";
        if(std::isfinite(value))out<<value;else out<<"null";
        out << '}';
    }
    out << "]}\n";
    NS_LOG_UNCOND("[AGORA] " << id << ' ' << state << " t=" << Simulator::Now().GetSeconds() << ' ' << reason);
    Snapshot();
}
std::string CacheName(uint16_t cell)
{
    for(auto& [label,dev]:g_cells) if(dev->GetCellId()==cell)
    {
        if(label=="5g")return "gNB_5G";
        if(label=="4g1")return "gNB_4G_1";
        if(label=="4g2")return "gNB_4G_2";
    }
    return "";
}
bool PrepareRet(const Change& c)
{
    if(g_lookups.empty())return true;
    if(g_retRoot.empty())return false;
    for(auto lookup:g_lookups)
        if(!lookup->PrepareRetCache(CacheName(c.cell),c.value,g_retRoot) ||
           !lookup->PrepareRetCache(CacheName(c.cell),c.expected,g_retRoot))return false;
    return true;
}
void Write(const Change& c, double value)
{
    auto dev=g_byId.at(c.cell);
    if(c.kind==1) for(uint32_t i=0;i<dev->GetCcMapSize();++i)dev->GetPhy(i)->SetTxPower(value);
    if(c.kind==2)
    {
        for(auto lookup:g_lookups)lookup->ActivateRetCache(CacheName(c.cell),value);
        for(uint32_t i=0;i<dev->GetCcMapSize();++i)
        {
            auto antenna=DynamicCast<UniformPlanarArray>(dev->GetPhy(i)->GetSpectrumPhy()->GetAntenna());
            antenna->SetAttribute("DowntiltAngle",DoubleValue(value*M_PI/180.0));
        }
    }
    if(c.kind==3)for(auto ue:Ues())ue->SetCellIndividualOffset(c.cell,value);
    if(c.kind==4 || c.kind==5)
    {
        auto cfg=dev->GetRrc()->GetA3HandoverConfig();
        dev->GetRrc()->UpdateA3HandoverConfig(c.kind==4?value:cfg.first,c.kind==5?value:cfg.second);
    }
}
void Restore(Command cmd)
{
    if(g_lease!=cmd.id){Event(cmd.id,"restore_skipped","lease_changed");return;}
    for(auto& c:cmd.changes)if(!Equal(Read(c.cell,c.kind),c.value))
    {
        g_lease.clear();Event(cmd.id,"restore_conflict","controls_changed_externally");return;
    }
    for(auto& c:cmd.changes)Write(c,c.expected);
    g_lease.clear();Event(cmd.id,"restored");
}
void Apply(Command cmd)
{
    for(auto& c:cmd.checks)if(!Equal(Read(c.cell,c.kind),c.expected))
    {
        g_lease.clear();Event(cmd.id,"rejected","baseline_changed_before_start");return;
    }
    for(auto& c:cmd.changes)Write(c,c.value);
    Event(cmd.id,"applied");
    Simulator::Schedule(Seconds(cmd.end-Simulator::Now().GetSeconds()),&Restore,cmd);
}
void Heartbeat()
{
    Snapshot();
    Simulator::Schedule(Seconds(1),&Heartbeat);
}
}

void AgoraRicController::RegisterLookup(Ptr<SionnaLookupChannelModel> lookup)
{
    g_lookups.push_back(lookup);
}
void AgoraRicController::InstallFromEnvironment(const std::map<std::string, Ptr<NrGnbNetDevice>>& cells,double stopTime)
{
    const char* dir=getenv("AGORA_CONTROL_DIR");
    const char* run=getenv("AGORA_RUN_ID");
    if(!dir || !run)return;
    NS_ABORT_MSG_IF(!SafeId(run),"AGORA_RUN_ID must be 32 lowercase hex characters");
    g_dir=std::string(dir)+"/"+run;g_run=run;g_stop=stopTime;g_cells=cells;
    g_lookups=SionnaLookupChannelModel::GetInstances();
    const char* replay=getenv("AGORA_REPLAY_RUN_ID");g_replay=replay?replay:"";
    NS_ABORT_MSG_IF(g_replay.find_first_not_of("abcdefghijklmnopqrstuvwxyz0123456789_")!=std::string::npos,"Invalid replay run id");
    const char* ret=getenv("AGORA_RET_CACHE_ROOT");g_retRoot=ret?ret:"";
    for(auto& [label,dev]:cells)
    {
        NS_ABORT_MSG_IF(label.find_first_not_of("abcdefghijklmnopqrstuvwxyz0123456789_")!=std::string::npos,"Invalid cell label");
        g_byId[dev->GetCellId()]=dev;
    }
    std::filesystem::create_directories(g_dir);
    NS_ABORT_MSG_IF(std::filesystem::exists(g_dir+"/status.json"),"AGORA run id already exists; use a new id");
    Simulator::Schedule(Seconds(0.01),&Heartbeat);
    Simulator::ScheduleDestroy([](){Snapshot(true);});
}
void AgoraRicController::Receive(const std::string& payload)
{
    if(g_dir.empty() || payload.size()>8192)return;
    std::istringstream input(payload);
    std::string version,run;Command cmd;size_t count=0;
    input>>version>>run>>cmd.id>>cmd.start>>cmd.end>>count;
    if(!input || version!="AGORA1" || !SafeId(cmd.id) || run!=g_run)return;
    if(auto it=g_seen.find(cmd.id);it!=g_seen.end())
    {Event(cmd.id,it->second==payload?"duplicate_ignored":"rejected","command_id_already_seen");return;}
    g_seen[cmd.id]=payload;
    if(!std::isfinite(cmd.start) || !std::isfinite(cmd.end) || cmd.start<=Simulator::Now().GetSeconds() ||
       cmd.end<=cmd.start || cmd.end>=g_stop || count==0 || count>15 || !g_lease.empty())
    {Event(cmd.id,"rejected","invalid_schedule_or_busy");return;}
    std::set<std::pair<uint16_t,int>> unique;
    for(size_t i=0;i<count;++i)
    {
        Change c{};input>>c.cell>>c.kind>>c.value>>c.expected;
        if(!input || !g_byId.count(c.cell) || !Domain(c.kind,c.value) || !Domain(c.kind,c.expected) ||
           !unique.emplace(c.cell,c.kind).second || !Equal(Read(c.cell,c.kind),c.expected))
        {Event(cmd.id,"rejected","invalid_control_or_baseline");return;}
        cmd.checks.push_back(c);
        if(c.kind==4 && !g_byId.at(c.cell)->GetRrc()->GetObject<NrGnbRrcProtocolIdeal>())
        {
            const std::set<int> values{0,40,64,80,100,128,160,256,320,480,512,640,1024,1280,2560,5120};
            if(!values.count(static_cast<int>(c.value)) || !values.count(static_cast<int>(c.expected)))
            {Event(cmd.id,"rejected","TTT_not_encodable_in_real_RRC");return;}
        }
        if(!Equal(c.value,c.expected))cmd.changes.push_back(c);
    }
    std::string extra;
    if(input>>extra || cmd.changes.empty()){Event(cmd.id,"rejected","empty_or_malformed_plan");return;}
    try
    {
        for(auto& c:cmd.changes)if(c.kind==2 && !PrepareRet(c))
        {Event(cmd.id,"rejected","matching_RET_cache_unavailable");return;}
    }
    catch(const std::exception&){Event(cmd.id,"rejected","RET_cache_load_failed");return;}
    g_lease=cmd.id;
    Event(cmd.id,"scheduled");
    Simulator::Schedule(Seconds(cmd.start-Simulator::Now().GetSeconds()),&Apply,cmd);
}
}

// SPDX-License-Identifier: GPL-2.0-only
#ifndef AGORA_RIC_CONTROLLER_H
#define AGORA_RIC_CONTROLLER_H
#include "ns3/ptr.h"
#include <map>
#include <string>

namespace ns3
{
class NrGnbNetDevice;
class SionnaLookupChannelModel;

/** Simulation-clock controller for the local AGORA E2SM-RC extension. */
class AgoraRicController
{
  public:
    /** Enable only when AGORA_CONTROL_DIR and AGORA_RUN_ID are provided. */
    static void InstallFromEnvironment(const std::map<std::string, Ptr<NrGnbNetDevice>>& cells,
                                       double stopTime);
    /** Register a lookup channel whose per-cell RET cache must follow controls. */
    static void RegisterLookup(Ptr<SionnaLookupChannelModel> lookup);
    /** Called on the simulator thread after decoding a real E2 control PDU. */
    static void Receive(const std::string& payload);
};
}
#endif

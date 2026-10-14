/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */
// Minimal, deterministic Energy_state (E2SM-RC style 300) control test.
//
// xapp_es_with_cell_util.c's autonomous "evaluate cells -> mark for
// shutdown -> hand UEs off -> switch off" state machine never completed a
// full cycle in live testing (see session notes). This xApp sidesteps that
// decision logic entirely: it waits a fixed delay (long enough for the
// scenario's own NrA3RsrpHandoverAlgorithm to have already moved the UE off
// the target cell -- no handover is sent here), then broadcasts a single
// Energy_state/Cell_Off RIC CONTROL message for a given cell to *every*
// connected E2 node, letting each node's own GetTargetCell()==m_cellId
// check (NrGnbNetDevice::ControlMessageReceivedCallback) decide whether to
// act on it. This also fixes what looks like a bug in
// switchOffCurrentCell(), which only ever sent to nodes.n[0] regardless of
// which node that actually was.
//
// The RAN-parameter encoding below (gen_rc_ctrl_hdr / gen_Target_Primary_
// Cell_ID / gen_cell_trigger_rc_ctrl_msg) is copied verbatim from
// xapp_es_with_cell_util.c's switchOffCurrentCell() path -- that encoding
// is not in question, only the decision logic that used to gate it.
//
// Env vars: WAIT_SECONDS (default 90), OFF_CELL (default 2).

#include "../../../../src/xApp/e42_xapp_api.h"
#include "../../../../src/sm/rc_sm/ie/ir/ran_param_struct.h"
#include "../../../../src/sm/rc_sm/ie/ir/ran_param_list.h"
#include "../../../../src/util/alg_ds/ds/lock_guard/lock_guard.h"
#include "../../../../src/sm/rc_sm/rc_sm_id.h"
#include "../../../../src/sm/rc_sm/ie/rc_data_ie.h"
#include "../../../../src/util/e.h"

#include <assert.h>
#include <stdlib.h>
#include <unistd.h>

typedef enum {
    ENERGY_STATE = 300
} rc_ctrl_service_style_id_e;

typedef enum {
    CELL_OFF = '0',
} cell_state_e;

static e2sm_rc_ctrl_hdr_frmt_1_t gen_rc_ctrl_hdr_frmt_1(ue_id_e2sm_t ue_id, uint32_t ric_style_type,
                                                         uint16_t ctrl_act_id)
{
    e2sm_rc_ctrl_hdr_frmt_1_t dst = {0};
    dst.ue_id = cp_ue_id_e2sm(&ue_id);
    dst.ric_style_type = ric_style_type;
    dst.ctrl_act_id = ctrl_act_id;
    return dst;
}

static e2sm_rc_ctrl_hdr_t gen_rc_ctrl_hdr(e2sm_rc_ctrl_hdr_e hdr_frmt, ue_id_e2sm_t ue_id,
                                          uint32_t ric_style_type, uint16_t ctrl_act_id)
{
    e2sm_rc_ctrl_hdr_t dst = {0};
    assert(hdr_frmt == FORMAT_1_E2SM_RC_CTRL_HDR && "not implemented the fill func for this ctrl hdr frmt");
    dst.format = FORMAT_1_E2SM_RC_CTRL_HDR;
    dst.frmt_1 = gen_rc_ctrl_hdr_frmt_1(ue_id, ric_style_type, ctrl_act_id);
    return dst;
}

static void set_EUTRA_CGI(seq_ran_param_t* EUTRA_CGI, const char targetcell)
{
    assert(EUTRA_CGI != NULL);
    assert(targetcell >= '0' && targetcell <= '9');

    if (EUTRA_CGI->ran_param_val.flag_false == NULL)
    {
        EUTRA_CGI->ran_param_val.flag_false = calloc(1, sizeof(ran_parameter_value_t));
        assert(EUTRA_CGI->ran_param_val.flag_false != NULL && "Memory exhausted");
    }
    EUTRA_CGI->ran_param_val.flag_false->type = BIT_STRING_RAN_PARAMETER_VALUE;

    byte_array_t target_ba = {0};
    target_ba.len = 1;
    target_ba.buf = malloc(sizeof(uint8_t));
    assert(target_ba.buf != NULL && "Memory exhausted");
    target_ba.buf[0] = targetcell;

    EUTRA_CGI->ran_param_val.flag_false->octet_str_ran.len = target_ba.len;
    EUTRA_CGI->ran_param_val.flag_false->octet_str_ran.buf = target_ba.buf;
}

static void gen_Target_Primary_Cell_ID(seq_ran_param_t* Target_Primary_Cell_ID, char targetcell)
{
    Target_Primary_Cell_ID->ran_param_id = TARGET_PRIMARY_CELL_ID_8_4_4_1;
    Target_Primary_Cell_ID->ran_param_val.type = STRUCTURE_RAN_PARAMETER_VAL_TYPE;
    Target_Primary_Cell_ID->ran_param_val.strct = calloc(1, sizeof(ran_param_struct_t));
    assert(Target_Primary_Cell_ID->ran_param_val.strct != NULL && "Memory exhausted");
    Target_Primary_Cell_ID->ran_param_val.strct->sz_ran_param_struct = 1;
    Target_Primary_Cell_ID->ran_param_val.strct->ran_param_struct = calloc(1, sizeof(seq_ran_param_t));
    assert(Target_Primary_Cell_ID->ran_param_val.strct->ran_param_struct != NULL && "Memory exhausted");

    seq_ran_param_t* CHOICE_Target_Cell = &Target_Primary_Cell_ID->ran_param_val.strct->ran_param_struct[0];
    CHOICE_Target_Cell->ran_param_id = CHOICE_TARGET_CELL_8_4_4_1;
    CHOICE_Target_Cell->ran_param_val.type = STRUCTURE_RAN_PARAMETER_VAL_TYPE;
    CHOICE_Target_Cell->ran_param_val.strct = calloc(1, sizeof(ran_param_struct_t));
    assert(CHOICE_Target_Cell->ran_param_val.strct != NULL && "Memory exhausted");
    CHOICE_Target_Cell->ran_param_val.strct->sz_ran_param_struct = 2;
    CHOICE_Target_Cell->ran_param_val.strct->ran_param_struct = calloc(2, sizeof(seq_ran_param_t));
    assert(CHOICE_Target_Cell->ran_param_val.strct->ran_param_struct != NULL && "Memory exhausted");

    seq_ran_param_t* NR_Cell = &CHOICE_Target_Cell->ran_param_val.strct->ran_param_struct[0];
    NR_Cell->ran_param_id = NR_CELL_8_4_4_1;
    NR_Cell->ran_param_val.type = STRUCTURE_RAN_PARAMETER_VAL_TYPE;
    NR_Cell->ran_param_val.strct = calloc(1, sizeof(ran_param_struct_t));
    assert(NR_Cell->ran_param_val.strct != NULL && "Memory exhausted");
    NR_Cell->ran_param_val.strct->sz_ran_param_struct = 1;
    NR_Cell->ran_param_val.strct->ran_param_struct = calloc(1, sizeof(seq_ran_param_t));

    seq_ran_param_t* NR_CGI = &NR_Cell->ran_param_val.strct->ran_param_struct[0];
    NR_CGI->ran_param_id = NR_CGI_8_4_4_1;
    NR_CGI->ran_param_val.type = ELEMENT_KEY_FLAG_FALSE_RAN_PARAMETER_VAL_TYPE;
    NR_CGI->ran_param_val.flag_false = calloc(1, sizeof(ran_parameter_value_t));
    assert(NR_CGI->ran_param_val.flag_false != NULL && "Memory exhausted");
    NR_CGI->ran_param_val.flag_false->type = BIT_STRING_RAN_PARAMETER_VALUE;
    char nr_cgi_str[1] = {targetcell};
    byte_array_t nr_cgi = cp_str_to_ba(nr_cgi_str);
    NR_CGI->ran_param_val.flag_false->octet_str_ran.len = nr_cgi.len;
    NR_CGI->ran_param_val.flag_false->octet_str_ran.buf = nr_cgi.buf;

    seq_ran_param_t* EUTRA_Cell = &CHOICE_Target_Cell->ran_param_val.strct->ran_param_struct[1];
    EUTRA_Cell->ran_param_id = EUTRA_CELL_8_4_4_1;
    EUTRA_Cell->ran_param_val.type = STRUCTURE_RAN_PARAMETER_VAL_TYPE;
    EUTRA_Cell->ran_param_val.strct = calloc(1, sizeof(ran_param_struct_t));
    assert(EUTRA_Cell->ran_param_val.strct != NULL && "Memory exhausted");
    EUTRA_Cell->ran_param_val.strct->sz_ran_param_struct = 1;
    EUTRA_Cell->ran_param_val.strct->ran_param_struct = calloc(1, sizeof(seq_ran_param_t));

    seq_ran_param_t* EUTRA_CGI = &EUTRA_Cell->ran_param_val.strct->ran_param_struct[0];
    EUTRA_CGI->ran_param_id = EUTRA_CGI_8_4_4_1;
    EUTRA_CGI->ran_param_val.type = ELEMENT_KEY_FLAG_FALSE_RAN_PARAMETER_VAL_TYPE;
    EUTRA_CGI->ran_param_val.flag_false = calloc(1, sizeof(ran_parameter_value_t));
    assert(EUTRA_CGI->ran_param_val.flag_false != NULL && "Memory exhausted");
    EUTRA_CGI->ran_param_val.flag_false->type = BIT_STRING_RAN_PARAMETER_VALUE;

    set_EUTRA_CGI(EUTRA_CGI, targetcell);
}

static e2sm_rc_ctrl_msg_frmt_1_t gen_rc_ctrl_msg_frmt_1_cell_trigger(char targetcell)
{
    e2sm_rc_ctrl_msg_frmt_1_t dst = {0};
    dst.sz_ran_param = 1;
    dst.ran_param = calloc(4, sizeof(seq_ran_param_t));
    assert(dst.ran_param != NULL && "Memory exhausted");
    gen_Target_Primary_Cell_ID(&dst.ran_param[0], targetcell);
    return dst;
}

static e2sm_rc_ctrl_msg_t gen_cell_trigger_rc_ctrl_msg(e2sm_rc_ctrl_msg_e msg_frmt, char targetcell)
{
    e2sm_rc_ctrl_msg_t dst = {0};
    assert(msg_frmt == FORMAT_1_E2SM_RC_CTRL_MSG && "not implemented the fill func for this ctrl msg frmt");
    dst.format = msg_frmt;
    dst.frmt_1 = gen_rc_ctrl_msg_frmt_1_cell_trigger(targetcell);
    return dst;
}

static ue_id_e2sm_t gen_rc_ue_id(ue_id_e2sm_e type, int ueid)
{
    ue_id_e2sm_t ue_id = {0};
    assert(type == GNB_UE_ID_E2SM && "not supported UE ID type");
    ue_id.type = GNB_UE_ID_E2SM;
    ue_id.gnb.ran_ue_id = (uint64_t*)malloc(sizeof(uint64_t));
    *(ue_id.gnb.ran_ue_id) = ueid;
    return ue_id;
}

int main(int argc, char* argv[])
{
    fr_args_t args = init_fr_args(argc, argv);
    init_xapp_api(&args);
    sleep(1);

    e2_node_arr_xapp_t nodes = e2_nodes_xapp_api();
    defer({ free_e2_node_arr_xapp(&nodes); });
    assert(nodes.len > 0);
    printf("[demo] Connected E2 nodes = %d\n", nodes.len);

    int wait_s = 90;
    const char* wait_env = getenv("WAIT_SECONDS");
    if (wait_env)
    {
        wait_s = atoi(wait_env);
    }
    int off_cell = 2;
    const char* cell_env = getenv("OFF_CELL");
    if (cell_env)
    {
        off_cell = atoi(cell_env);
    }

    printf("[demo] Waiting %d seconds (for the scenario's own handover algorithm to "
           "vacate cell %d) before sending Cell_Off...\n",
           wait_s, off_cell);
    sleep((unsigned int)wait_s);

    ue_id_e2sm_t ue_id = gen_rc_ue_id(GNB_UE_ID_E2SM, 1);
    rc_ctrl_req_data_t rc_ctrl = {0};
    rc_ctrl.hdr = gen_rc_ctrl_hdr(FORMAT_1_E2SM_RC_CTRL_HDR, ue_id, ENERGY_STATE, CELL_OFF);
    char targetCellChar = (char)('0' + off_cell);
    rc_ctrl.msg = gen_cell_trigger_rc_ctrl_msg(FORMAT_1_E2SM_RC_CTRL_MSG, targetCellChar);

    printf("[demo] Broadcasting Energy_state/Cell_Off for cell %d to all %d connected node(s)\n",
           off_cell, nodes.len);
    for (size_t i = 0; i < nodes.len; ++i)
    {
        sm_ans_xapp_t ans = control_sm_xapp_api(&nodes.n[i].id, SM_RC_ID, &rc_ctrl);
        printf("[demo]   node %zu: %s\n", i, ans.success ? "sent OK" : "send FAILED");
    }
    free_rc_ctrl_req_data(&rc_ctrl);

    printf("[demo] Done. Sleeping 5s then exiting.\n");
    sleep(5);

    while (try_stop_xapp_api() == false)
    {
        usleep(1000);
    }

    printf("[demo] xApp completed.\n");
    return 0;
}

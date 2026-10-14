/* SPDX-License-Identifier: LicenseRef-OAI-Public-License-1.1
 * Local AGORA E2SM-RC extension: style 401, action 1, parameter 33000.
 * A successful send is NOT an ns-3 application receipt.
 */
#include "../../../../src/xApp/e42_xapp_api.h"
#include "../../../../src/sm/rc_sm/rc_sm_id.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int main(int argc, char** argv)
{
  const char* payload = getenv("AGORA_E2_PAYLOAD");
  const char* target = getenv("AGORA_E2_NODE_ID");
  if (!payload || !target || strlen(payload) > 8192 || strncmp(payload, "AGORA1 ", 7)) {
    fprintf(stderr, "AGORA_E2_PAYLOAD and AGORA_E2_NODE_ID are required\n");
    return 2;
  }
  unsigned long node_id = strtoul(target, NULL, 10);
  fr_args_t args = init_fr_args(argc, argv);
  init_xapp_api(&args);
  e2_node_arr_xapp_t nodes = {0};
  int selected = -1;
  for (int attempt = 0; attempt < 50 && selected < 0; ++attempt) {
    free_e2_node_arr_xapp(&nodes);
    nodes = e2_nodes_xapp_api();
    for (int i = 0; i < nodes.len; ++i) {
      if (attempt == 0) fprintf(stderr,"AGORA_E2_NODE index=%d id=%u\n",i,nodes.n[i].id.nb_id.nb_id);
      if (nodes.n[i].id.nb_id.nb_id == node_id) selected = i;
    }
    if (selected < 0) usleep(100000);
  }
  if (selected < 0) {
    fprintf(stderr, "Requested E2 node is not connected\n");
    return 3;
  }
  rc_ctrl_req_data_t request = {0};
  request.hdr.format = FORMAT_1_E2SM_RC_CTRL_HDR;
  request.hdr.frmt_1.ric_style_type = 401;
  request.hdr.frmt_1.ctrl_act_id = 1;
  request.hdr.frmt_1.ue_id.type = GNB_UE_ID_E2SM;
  request.hdr.frmt_1.ue_id.gnb.guami.plmn_id.mcc = 1;
  request.hdr.frmt_1.ue_id.gnb.guami.plmn_id.mnc = 1;
  request.hdr.frmt_1.ue_id.gnb.guami.plmn_id.mnc_digit_len = 2;
  request.msg.format = FORMAT_1_E2SM_RC_CTRL_MSG;
  request.msg.frmt_1.sz_ran_param = 1;
  request.msg.frmt_1.ran_param = calloc(1, sizeof(seq_ran_param_t));
  assert(request.msg.frmt_1.ran_param);
  seq_ran_param_t* param = request.msg.frmt_1.ran_param;
  param->ran_param_id = 33000;
  param->ran_param_val.type = ELEMENT_KEY_FLAG_FALSE_RAN_PARAMETER_VAL_TYPE;
  param->ran_param_val.flag_false = calloc(1, sizeof(ran_parameter_value_t));
  assert(param->ran_param_val.flag_false);
  param->ran_param_val.flag_false->type = OCTET_STRING_RAN_PARAMETER_VALUE;
  param->ran_param_val.flag_false->octet_str_ran.len = strlen(payload);
  param->ran_param_val.flag_false->octet_str_ran.buf = (uint8_t*)strdup(payload);
  sm_ans_xapp_t result = control_sm_xapp_api(&nodes.n[selected].id, SM_RC_ID, &request);
  printf("AGORA_E2_SUBMITTED node=%lu accepted_by_xapp=%d\n", node_id, result.success);
  fflush(stdout);
  free_rc_ctrl_req_data(&request);
  sleep(1);
  free_e2_node_arr_xapp(&nodes);
  for (int i=0; i<30 && !try_stop_xapp_api(); ++i) usleep(100000);
  return result.success ? 0 : 4;
}

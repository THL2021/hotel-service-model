#!/bin/bash
package_name="cj_model_lib.tar.gz"
algo_name="nrm_mtl_v1_seq"

sudo rm -rf ${package_name}

sudo tar czf ./${package_name} *

echo "package success"

/Users/hltan_2017/Programs/odps_client/bin/odpscmd -e "add file ${package_name} as cj_model_lib_${algo_name}.tar.gz -f;"

echo "update odps package success"

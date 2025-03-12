function eval_mask_acc {
    local config=$1
    local ckpt=$2
    local ngpu=$3
    local tag=$4
    local use_psm=$5
    shift 5
    local opts=$@

    echo "config: $config"
    echo "ckpt: $ckpt"
    echo "ngpu: $ngpu"
    echo "tag: $tag"
    echo "use_psm: $use_psm"
    echo "opts: $opts"

    if [ $use_psm -eq 1 ]; then
        python train_maskclippp.py \
        --config-file $config \
        --num-gpus $ngpu \
        --tag $tag \
        --eval-only \
        --dist-url "auto" \
        MODEL.WEIGHTS $ckpt \
        MODEL.MASKCLIPPP.TEST.ENSEMBLE_ON False \
        MODEL.MASKCLIPPP.TEST.MASK_ACC True \
        MODEL.MASKCLIPPP.SEGMENTOR.NAME GTSegmentor \
        MODEL.MASKCLIPPP.VISUAL_ENCODER_F.NAME DummyVisualEncoder \
        MODEL.MASKCLIPPP.TEXT_ENCODER_F.NAME none \
        $opts
    else
    python train_maskclippp.py \
        --config-file $config \
        --num-gpus $ngpu \
        --tag $tag \
        --eval-only \
        --dist-url "auto" \
        MODEL.WEIGHTS $ckpt \
        MODEL.MASKCLIPPP.TEST.ENSEMBLE_ON False \
        MODEL.MASKCLIPPP.TEST.MASK_ACC True \
        MODEL.MASKCLIPPP.SEGMENTOR.NAME GTSegmentor \
        MODEL.MASKCLIPPP.VISUAL_ENCODER_F.NAME DummyVisualEncoder \
        MODEL.MASKCLIPPP.TEXT_ENCODER_F.NAME none \
        MODEL.MASKCLIPPP.PSM.NAME "DummyPSM" \
        MODEL.MASKCLIPPP.USE_LOGIT_SCALE True \
        $opts
    fi
}


function eval_pan_mask_acc {
    local config=$1
    local ckpt=$2
    local ngpu=$3
    local tag=$4
    local use_psm=$5
    shift 5
    local opts=$@

    echo "config: $config"
    echo "ckpt: $ckpt"
    echo "ngpu: $ngpu"
    echo "tag: $tag"
    echo "use_psm: $use_psm"
    echo "opts: $opts"

    if [ $use_psm -eq 1 ]; then
        python train_maskclippp.py \
        --config-file $config \
        --num-gpus $ngpu \
        --tag $tag \
        --eval-only \
        --dist-url "auto" \
        MODEL.WEIGHTS $ckpt \
        MODEL.MASKCLIPPP.TEST.ENSEMBLE_ON False \
        MODEL.MASKCLIPPP.TEST.MASK_ACC True \
        MODEL.MASKCLIPPP.SEGMENTOR.NAME PanGTSegmentor \
        MODEL.MASKCLIPPP.VISUAL_ENCODER_F.NAME DummyVisualEncoder \
        MODEL.MASKCLIPPP.TEXT_ENCODER_F.NAME none \
        $opts
    else
    python train_maskclippp.py \
        --config-file $config \
        --num-gpus $ngpu \
        --tag $tag \
        --eval-only \
        --dist-url "auto" \
        MODEL.WEIGHTS $ckpt \
        MODEL.MASKCLIPPP.TEST.ENSEMBLE_ON False \
        MODEL.MASKCLIPPP.TEST.MASK_ACC True \
        MODEL.MASKCLIPPP.SEGMENTOR.NAME PanGTSegmentor \
        MODEL.MASKCLIPPP.VISUAL_ENCODER_F.NAME DummyVisualEncoder \
        MODEL.MASKCLIPPP.TEXT_ENCODER_F.NAME none \
        MODEL.MASKCLIPPP.PSM.NAME "DummyPSM" \
        MODEL.MASKCLIPPP.USE_LOGIT_SCALE True \
        $opts
    fi
}


function eval_mask_acc_ade150 {
    local config=$1
    local ckpt=$2
    local ngpu=$3
    local tag=$4_ade150
    local use_psm=$5
    shift 5
    local opts=$@
    eval_mask_acc $config $ckpt $ngpu $tag $use_psm $opts
}

function eval_mask_acc_stuff {
    local config=$1
    local ckpt=$2
    local ngpu=$3
    local tag=$4_stuff
    local use_psm=$5
    shift 5
    local opts=$@
    eval_mask_acc $config $ckpt $ngpu $tag $use_psm $opts \
        DATASETS.TEST "(\"openvocab_coco_2017_val_stuff_sem_seg\",)"
}

function eval_pan_mask_acc_coco {
    local config=$1
    local ckpt=$2
    local ngpu=$3
    local tag=$4_coco
    local use_psm=$5
    shift 5
    local opts=$@
    eval_pan_mask_acc $config $ckpt $ngpu $tag $use_psm $opts \
        DATASETS.TEST "(\"openvocab_coco_2017_val_panoptic_with_sem_seg\",)"
}


function eval_mask_acc_ade847 {
    local config=$1
    local ckpt=$2
    local ngpu=$3
    local tag=$4_ade847
    local use_psm=$5
    shift 5
    local opts=$@
    eval_mask_acc $config $ckpt $ngpu $tag $use_psm $opts \
        DATASETS.TEST "(\"openvocab_ade20k_full_sem_seg_val\",)"
}

function eval_mask_acc_ctx459 {
    local config=$1
    local ckpt=$2
    local ngpu=$3
    local tag=$4_ctx459
    local use_psm=$5
    shift 5
    local opts=$@
    eval_mask_acc $config $ckpt $ngpu $tag $use_psm $opts \
        DATASETS.TEST "(\"openvocab_pascal_ctx459_sem_seg_val\",)"
}

function eval_mask_acc_ctx59 {
    local config=$1
    local ckpt=$2
    local ngpu=$3
    local tag=$4_ctx59
    local use_psm=$5
    shift 5
    local opts=$@
    eval_mask_acc $config $ckpt $ngpu $tag $use_psm $opts \
        DATASETS.TEST "(\"openvocab_pascal_ctx59_sem_seg_val\",)"
}

function eval_mask_acc_pc20 {
    local config=$1
    local ckpt=$2
    local ngpu=$3
    local tag=$4_pc20
    local use_psm=$5
    shift 5
    local opts=$@
    eval_mask_acc $config $ckpt $ngpu $tag $use_psm $opts \
        DATASETS.TEST "(\"openvocab_pascal20_sem_seg_val\",)"
}


function eval_mask_acc_city {
    local config=$1
    local ckpt=$2
    local ngpu=$3
    local tag=$4_city
    local use_psm=$5
    shift 5
    local opts=$@
    eval_mask_acc $config $ckpt $ngpu $tag $use_psm $opts \
        DATASETS.TEST "(\"openvocab_cityscapes_fine_panoptic_val\",)"
}


function eval_mask_acc_dark_zurich {
    local config=$1
    local ckpt=$2
    local ngpu=$3
    local tag=$4_dark_zurich
    local use_psm=$5
    shift 5
    local opts=$@
    eval_mask_acc $config $ckpt $ngpu $tag $use_psm $opts \
        DATASETS.TEST "(\"dark_zurich_sem_seg_val\",)"
}

function eval_mask_acc_foodseg {
    local config=$1
    local ckpt=$2
    local ngpu=$3
    local tag=$4_foodseg
    local use_psm=$5
    shift 5
    local opts=$@
    eval_mask_acc $config $ckpt $ngpu $tag $use_psm $opts \
        DATASETS.TEST "(\"foodseg103_sem_seg_test\",)"
}

function eval_mask_acc_mhp {
    local config=$1
    local ckpt=$2
    local ngpu=$3
    local tag=$4_mhp
    local use_psm=$5
    shift 5
    local opts=$@
    eval_mask_acc $config $ckpt $ngpu $tag $use_psm $opts \
        DATASETS.TEST "(\"mhp_v1_sem_seg_test\",)"
}

function eval_mask_acc_kvasir {
    local config=$1
    local ckpt=$2
    local ngpu=$3
    local tag=$4_kvasir
    local use_psm=$5
    shift 5
    local opts=$@
    eval_mask_acc $config $ckpt $ngpu $tag $use_psm $opts \
        DATASETS.TEST "(\"kvasir_instrument_sem_seg_test\",)"
}

function eval_mask_acc_pst900 {
    local config=$1
    local ckpt=$2
    local ngpu=$3
    local tag=$4_pst900
    local use_psm=$5
    shift 5
    local opts=$@
    eval_mask_acc $config $ckpt $ngpu $tag $use_psm $opts \
        DATASETS.TEST "(\"pst900_sem_seg_test\",)"
}


function eval_mask_acc_corrosion {
    local config=$1
    local ckpt=$2
    local ngpu=$3
    local tag=$4_corrosion
    local use_psm=$5
    shift 5
    local opts=$@
    eval_mask_acc $config $ckpt $ngpu $tag $use_psm $opts \
        DATASETS.TEST "(\"corrosion_cs_sem_seg_test\",)"
}

function eval_mask_acc_zerowaste {
    local config=$1
    local ckpt=$2
    local ngpu=$3
    local tag=$4_zerowaste
    local use_psm=$5
    shift 5
    local opts=$@
    eval_mask_acc $config $ckpt $ngpu $tag $use_psm $opts \
        DATASETS.TEST "(\"zerowaste_sem_seg_test\",)"
}


function eval_mask_acc_cub200 {
    local config=$1
    local ckpt=$2
    local ngpu=$3
    local tag=$4_cub200
    local use_psm=$5
    shift 5
    local opts=$@
    eval_mask_acc $config $ckpt $ngpu $tag $use_psm $opts \
        DATASETS.TEST "(\"cub_200_sem_seg_test\",)"
}


function eval_mask_acc_bdd100k {
    local config=$1
    local ckpt=$2
    local ngpu=$3
    local tag=$4_bdd100k
    local use_psm=$5
    shift 5
    local opts=$@
    eval_mask_acc $config $ckpt $ngpu $tag $use_psm $opts \
        DATASETS.TEST "(\"bdd100k_sem_seg_val\",)"
}

function eval_mask_acc_atlantis {
    local config=$1
    local ckpt=$2
    local ngpu=$3
    local tag=$4_atlantis
    local use_psm=$5
    shift 5
    local opts=$@
    eval_mask_acc $config $ckpt $ngpu $tag $use_psm $opts \
        DATASETS.TEST "(\"atlantis_sem_seg_test\",)"
}

function eval_mask_acc_dram {
    local config=$1
    local ckpt=$2
    local ngpu=$3
    local tag=$4_dram
    local use_psm=$5
    shift 5
    local opts=$@
    eval_mask_acc $config $ckpt $ngpu $tag $use_psm $opts \
        DATASETS.TEST "(\"dram_sem_seg_test\",)"
}

function eval_mask_acc_isaid {
    local config=$1
    local ckpt=$2
    local ngpu=$3
    local tag=$4_isaid
    local use_psm=$5
    shift 5
    local opts=$@
    eval_mask_acc $config $ckpt $ngpu $tag $use_psm $opts \
        DATASETS.TEST "(\"isaid_sem_seg_val\",)"
}

function eval_mask_acc_chase_db1 {
    local config=$1
    local ckpt=$2
    local ngpu=$3
    local tag=$4_chase_db1
    local use_psm=$5
    shift 5
    local opts=$@
    eval_mask_acc $config $ckpt $ngpu $tag $use_psm $opts \
        DATASETS.TEST "(\"chase_db1_sem_seg_test\",)"
}


function eval_mask_acc_suim {
    local config=$1
    local ckpt=$2
    local ngpu=$3
    local tag=$4_suim
    local use_psm=$5
    shift 5
    local opts=$@
    eval_mask_acc $config $ckpt $ngpu $tag $use_psm $opts \
        DATASETS.TEST "(\"suim_sem_seg_test\",)"
}


function eval_mask_acc_paxray {
    local config=$1
    local ckpt=$2
    local ngpu=$3
    local tag=$4_paxray
    local use_psm=$5
    shift 5
    local opts=$@
    eval_mask_acc $config $ckpt $ngpu $tag $use_psm $opts \
        DATASETS.TEST "(\"paxray_sem_seg_test_lungs\",)"
}


function eval_mask_acc_isprs {
    local config=$1
    local ckpt=$2
    local ngpu=$3
    local tag=$4_isprs
    local use_psm=$5
    shift 5
    local opts=$@
    eval_mask_acc $config $ckpt $ngpu $tag $use_psm $opts \
        DATASETS.TEST "(\"isprs_potsdam_sem_seg_test_irrg\",)"
}

function eval_mask_acc_cryonuseg {
    local config=$1
    local ckpt=$2
    local ngpu=$3
    local tag=$4_cryonuseg
    local use_psm=$5
    shift 5
    local opts=$@
    eval_mask_acc $config $ckpt $ngpu $tag $use_psm $opts \
        DATASETS.TEST "(\"cryonuseg_sem_seg_test\",)"
}

function eval_mask_acc_deepcrack {
    local config=$1
    local ckpt=$2
    local ngpu=$3
    local tag=$4_deepcrack
    local use_psm=$5
    shift 5
    local opts=$@
    eval_mask_acc $config $ckpt $ngpu $tag $use_psm $opts \
        DATASETS.TEST "(\"deepcrack_sem_seg_test\",)"
}

function eval_mask_acc_floodnet {
    local config=$1
    local ckpt=$2
    local ngpu=$3
    local tag=$4_floodnet
    local use_psm=$5
    shift 5
    local opts=$@
    eval_mask_acc $config $ckpt $ngpu $tag $use_psm $opts \
        DATASETS.TEST "(\"floodnet_sem_seg_test\",)"
}

function eval_mask_acc_cwfid {
    local config=$1
    local ckpt=$2
    local ngpu=$3
    local tag=$4_cwfid
    local use_psm=$5
    shift 5
    local opts=$@
    eval_mask_acc $config $ckpt $ngpu $tag $use_psm $opts \
        DATASETS.TEST "(\"cwfid_sem_seg_test\",)"
}


function eval_mask_acc_worldfloods {
    local config=$1
    local ckpt=$2
    local ngpu=$3
    local tag=$4_worldfloods
    local use_psm=$5
    shift 5
    local opts=$@
    eval_mask_acc $config $ckpt $ngpu $tag $use_psm $opts \
        DATASETS.TEST "(\"worldfloods_sem_seg_test_irrg\",)"
}

function eval_mask_acc_uavid {
    local config=$1
    local ckpt=$2
    local ngpu=$3
    local tag=$4_uavid
    local use_psm=$5
    shift 5
    local opts=$@
    eval_mask_acc $config $ckpt $ngpu $tag $use_psm $opts \
        DATASETS.TEST "(\"uavid_sem_seg_val\",)"
}


function eval_mask_acc_base {
    eval_mask_acc_ade150 $@ && \
    eval_mask_acc_ade847 $@ && \
    eval_mask_acc_ctx459 $@ && \
    eval_mask_acc_ctx59 $@ && \
    eval_mask_acc_stuff $@
}
function eval_default {
    config=$1
    ckpt=$2
    ngpu=$3
    tag=$4
    shift 4
    opts=$@

    echo "config: $config"
    echo "ckpt: $ckpt"
    echo "ngpu: $ngpu"
    echo "tag: $tag"
    echo "opts: $opts"

    python train_maskclippp.py \
        --config-file $config \
        --num-gpus $ngpu \
        --tag $tag \
        --eval-only \
        --dist-url "auto" \
        MODEL.WEIGHTS $ckpt \
        $opts
}

function eval_without_psm {
    config=$1
    ckpt=$2
    ngpu=$3
    tag=$4
    shift 4
    opts=$@

    echo "config: $config"
    echo "ckpt: $ckpt"
    echo "ngpu: $ngpu"
    echo "tag: $tag"
    echo "opts: $opts"

    python train_maskclippp.py \
        --config-file $config \
        --num-gpus $ngpu \
        --tag ${tag}_without-psm \
        --eval-only \
        --dist-url "auto" \
        MODEL.WEIGHTS $ckpt \
        MODEL.MASKCLIPPP.PSM.NAME "DummyPSM" \
        MODEL.MASKCLIPPP.USE_LOGIT_SCALE True \
        $opts
}

function eval_ade150 {
    config=$1
    ckpt=$2
    ngpu=$3
    tag=$4
    shift 4
    opts=$@

    echo "config: $config"
    echo "ckpt: $ckpt"
    echo "ngpu: $ngpu"
    echo "tag: $tag" 
    echo "opts: $opts"

    python train_maskclippp.py \
        --config-file $config \
        --num-gpus $ngpu \
        --tag ${tag}_ade150 \
        --eval-only \
        --dist-url "auto" \
        MODEL.WEIGHTS $ckpt \
        DATASETS.TEST "(\"openvocab_ade20k_panoptic_val\",)" \
        $opts

}

function eval_ade847 {
    config=$1
    ckpt=$2
    ngpu=$3
    tag=$4
    shift 4
    opts=$@

    echo "config: $config"
    echo "ckpt: $ckpt"
    echo "ngpu: $ngpu"
    echo "tag: $tag"
    echo "opts: $opts"

    python train_maskclippp.py \
        --config-file $config \
        --num-gpus $ngpu \
        --tag ${tag}_ade847 \
        --eval-only \
        --dist-url "auto" \
        MODEL.WEIGHTS $ckpt \
        DATASETS.TEST "(\"openvocab_ade20k_full_sem_seg_val\",)" \
        MODEL.MASK_FORMER.TEST.PANOPTIC_ON False \
        MODEL.MASK_FORMER.TEST.INSTANCE_ON False \
        $opts

}

function eval_pc20 {
    config=$1
    ckpt=$2
    ngpu=$3
    tag=$4
    shift 4
    opts=$@

    echo "config: $config"
    echo "ckpt: $ckpt"
    echo "ngpu: $ngpu"
    echo "tag: $tag"
    echo "opts: $opts"

    python train_maskclippp.py \
        --config-file $config \
        --num-gpus $ngpu \
        --tag ${tag}_pc20 \
        --eval-only \
        --dist-url "auto" \
        MODEL.WEIGHTS $ckpt \
        MODEL.MASK_FORMER.TEST.PANOPTIC_ON False \
        MODEL.MASK_FORMER.TEST.INSTANCE_ON False \
        DATASETS.TEST "(\"openvocab_pascal20_sem_seg_val\",)" \
        $opts

}


function eval_ctx59 {
    config=$1
    ckpt=$2
    ngpu=$3
    tag=$4
    shift 4
    opts=$@

    echo "config: $config"
    echo "ckpt: $ckpt"
    echo "ngpu: $ngpu"
    echo "tag: $tag"
    echo "opts: $opts"

    python train_maskclippp.py \
        --config-file $config \
        --num-gpus $ngpu \
        --tag ${tag}_ctx59 \
        --eval-only \
        --dist-url "auto" \
        MODEL.WEIGHTS $ckpt \
        DATASETS.TEST "(\"openvocab_pascal_ctx59_sem_seg_val\",)" \
        MODEL.MASK_FORMER.TEST.PANOPTIC_ON False \
        MODEL.MASK_FORMER.TEST.INSTANCE_ON False \
        $opts

}

function eval_ctx459 {
    config=$1
    ckpt=$2
    ngpu=$3
    tag=$4
    shift 4
    opts=$@

    echo "config: $config"
    echo "ckpt: $ckpt"
    echo "ngpu: $ngpu"
    echo "tag: $tag"
    echo "opts: $opts"

    python train_maskclippp.py \
        --config-file $config \
        --num-gpus $ngpu \
        --tag ${tag}_ctx459 \
        --eval-only \
        --dist-url "auto" \
        MODEL.WEIGHTS $ckpt \
        DATASETS.TEST "(\"openvocab_pascal_ctx459_sem_seg_val\",)" \
        MODEL.MASK_FORMER.TEST.PANOPTIC_ON False \
        MODEL.MASK_FORMER.TEST.INSTANCE_ON False \
        $opts

}

function eval_coco133 {
    config=$1
    ckpt=$2
    ngpu=$3
    tag=$4
    shift 4
    opts=$@

    echo "config: $config"
    echo "ckpt: $ckpt"
    echo "ngpu: $ngpu"
    echo "tag: $tag"
    echo "opts: $opts"

    python train_maskclippp.py \
        --config-file $config \
        --num-gpus $ngpu \
        --tag ${tag}_coco133 \
        --eval-only \
        --dist-url "auto" \
        MODEL.WEIGHTS $ckpt \
        DATASETS.TEST "(\"openvocab_coco_2017_val_panoptic_with_sem_seg\",)" \
        INPUT.MIN_SIZE_TEST 1024 \
        INPUT.MAX_SIZE_TEST 2560
        $opts

}


function eval_cocostuff {
    config=$1
    ckpt=$2
    ngpu=$3
    tag=$4
    shift 4
    opts=$@

    echo "config: $config"
    echo "ckpt: $ckpt"
    echo "ngpu: $ngpu"
    echo "tag: $tag"
    echo "opts: $opts"

    python train_maskclippp.py \
        --config-file $config \
        --num-gpus $ngpu \
        --tag ${tag}_cocostuff \
        --eval-only \
        --dist-url "auto" \
        MODEL.WEIGHTS $ckpt \
        DATASETS.TEST "(\"openvocab_coco_2017_val_stuff_sem_seg\",)" \
        MODEL.MASK_FORMER.TEST.PANOPTIC_ON False \
        MODEL.MASK_FORMER.TEST.INSTANCE_ON False \
        $opts
}



function eval_all {
    eval_ade150 $@
    eval_ade847 $@ && \
    eval_ctx459 $@ && \
    eval_ctx59 $@ && \
    eval_pc20 $@
}
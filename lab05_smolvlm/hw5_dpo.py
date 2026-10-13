"""实验五：合成图像、NF4 QLoRA SFT、手写 DPO、三阶段及 beta 对照。

指南的 15 题既用于 SFT 又用于评测，因此这里报告的是同题回测，不是独立测试泛化。
数据、训练顺序、步数和判分口径在第一次推理前固定，不根据结果修改。
"""
import argparse
import csv
import json
import math
import os
import platform
import random
import re
import time
from pathlib import Path

MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"
ROOT = Path(os.environ.get("LAB05_ROOT", Path(__file__).resolve().parents[1]))
OUT = ROOT / "outputs"
IMG_DIR = OUT / "images"
SEED = 42
COLORS = {"红": (220, 50, 50), "蓝": (50, 80, 220),
          "绿": (40, 170, 70), "黄": (230, 190, 40)}
IMG_SPEC = {
    "img01": [("红", "圆", 2), ("蓝", "方", 1)],
    "img02": [("绿", "圆", 3)],
    "img03": [("蓝", "圆", 1), ("黄", "方", 2)],
    "img04": [("红", "方", 3)],
    "img05": [("绿", "方", 2), ("红", "圆", 1)],
    "img06": [("黄", "圆", 4)],
    "img07": [("蓝", "方", 2), ("绿", "圆", 1)],
    "img08": [("红", "圆", 4)],
    "img09": [("绿", "方", 1), ("黄", "圆", 2)],
    "img10": [("蓝", "圆", 3)],
    "img11": [("黄", "方", 1)],
    "img12": [("红", "方", 1), ("蓝", "方", 1)],
    "img13": [("绿", "圆", 2), ("黄", "圆", 1)],
    "img14": [("蓝", "方", 3), ("红", "方", 1)],
    "img15": [("黄", "圆", 2), ("绿", "圆", 2)],
}

def question(qid, typ, img, prompt, gold):
    return dict(id=qid, type=typ, img=img, q=prompt, gold=gold)

QUESTIONS = [
    question("P01", "count", "img01", "图中有几个红色圆形？", "2"),
    question("P02", "count", "img02", "图中有几个绿色圆形？", "3"),
    question("P03", "count", "img03", "图中有几个黄色方形？", "2"),
    question("P04", "count", "img06", "图中有几个黄色圆形？", "4"),
    question("P05", "count", "img10", "图中有几个蓝色圆形？", "3"),
    question("P06", "attr", "img01", "图中蓝色的图形是什么形状？", "方形"),
    question("P07", "attr", "img05", "图中红色的图形是什么形状？", "圆形"),
    question("P08", "attr", "img09", "图中绿色的图形是什么形状？", "方形"),
    question("P09", "attr", "img13", "图中黄色的图形是什么形状？", "圆形"),
    question("P10", "attr", "img14", "图中蓝色的图形是什么形状？", "方形"),
    question("P11", "exist", "img11", "图里有没有黄色方形？", "有"),
    question("P12", "exist", "img12", "图里有没有红色圆形？", "没有"),
    question("P13", "exist", "img07", "图里有没有绿色圆形？", "有"),
    question("P14", "exist", "img08", "图里有没有绿色方形？", "没有"),
    question("P15", "exist", "img15", "图里有没有蓝色图形？", "没有"),
]
SFT_DATA = [dict(img=q["img"], prompt=q["q"], answer=q["gold"] +
                 ("个。" if q["type"] == "count" else "。")) for q in QUESTIONS]
REJECTED = {
    "P01": "3 个。图中有三个红色圆形整齐排列。", "P02": "2 个。",
    "P03": "没有看到黄色图形。", "P04": "3 个。", "P05": "图中有三个蓝色方形。",
    "P06": "圆形。", "P07": "方形，而且有两个红色方形。", "P08": "图中没有绿色图形。",
    "P11": "没有。", "P12": "有，图左上角有一个红色圆形。",
    "P14": "有，一共两个绿色方形。", "P15": "有，两个蓝色圆形。",
}
CHOSEN = {"P01":"2 个。", "P02":"3 个。", "P03":"2 个。", "P04":"4 个。",
          "P05":"3 个。", "P06":"方形。", "P07":"圆形。", "P08":"方形。",
          "P11":"有。", "P12":"没有。", "P14":"没有。",
          "P15":"没有。图中只有黄色和绿色的圆形。"}
PRE_PAIRS = [dict(id=q["id"], img=q["img"], prompt=q["q"], chosen=CHOSEN[q["id"]],
                  rejected=REJECTED[q["id"]], type=q["type"]) for q in QUESTIONS if q["id"] in REJECTED]
EXTRA_QUESTIONS = [
    question("E01", "count", "img04", "图中有几个红色方形？", "3"),
    question("E02", "attr", "img08", "图中红色的图形是什么形状？", "圆形"),
    question("E03", "exist", "img13", "图里有没有蓝色方形？", "没有"),
    question("E04", "exist", "img14", "图里有没有红色方形？", "有"),
    question("E05", "exist", "img15", "图里有没有黄色圆形？", "有"),
]
EXTRA_PAIRS = [
    dict(id="E01", img="img04", prompt=EXTRA_QUESTIONS[0]["q"], chosen="3 个。",
         rejected="2 个。", type="count", failure_mode="数错"),
    dict(id="E02", img="img08", prompt=EXTRA_QUESTIONS[1]["q"], chosen="圆形。",
         rejected="方形。", type="attr", failure_mode="形状幻觉"),
    dict(id="E03", img="img13", prompt=EXTRA_QUESTIONS[2]["q"], chosen="没有。",
         rejected="有，图中有两个蓝色方形。", type="exist", failure_mode="不存在物体硬答有"),
    dict(id="E04", img="img14", prompt=EXTRA_QUESTIONS[3]["q"], chosen="有。",
         rejected="没有，图中没有红色图形。", type="exist", failure_mode="存在物体误拒答"),
    dict(id="E05", img="img15", prompt=EXTRA_QUESTIONS[4]["q"], chosen="有。图中有两个黄色圆形。",
         rejected="有。图中还有两个蓝色圆形和一只猫。", type="exist", failure_mode="答案有但理由编造"),
]

def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")

def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def make_data():
    from PIL import Image, ImageDraw
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    rnd = random.Random(SEED)
    boxes = {}
    for img_id, spec in IMG_SPEC.items():
        im = Image.new("RGB", (256, 256), "white")
        draw = ImageDraw.Draw(im)
        boxes[img_id] = []
        for color, shape, n in spec:
            for _ in range(n):
                x, y, s = rnd.randint(10, 200), rnd.randint(10, 200), rnd.randint(30, 50)
                box = [x, y, x+s, y+s]
                (draw.ellipse if shape == "圆" else draw.rectangle)(box, outline=COLORS[color], width=4)
                boxes[img_id].append(dict(color=color, shape=shape, box=box))
        path = IMG_DIR / f"{img_id}.png"
        if not path.exists():
            im.save(path)
    write_json(OUT/"data_manifest.json", dict(seed=SEED, image_spec=IMG_SPEC, boxes=boxes,
        questions=QUESTIONS, sft_data=SFT_DATA, pre_pairs=PRE_PAIRS, extra_pairs=EXTRA_PAIRS,
        evaluation_note="15 题同题回测；新增 5 题在增强 DPO 中参与训练，均非独立测试集。"))
    # 保留原指南随机位置，可能重叠；不私自重绘成更容易的图片。
    fig = Image.new("RGB", (5*256, 3*286), "white")
    d = ImageDraw.Draw(fig)
    for i, img_id in enumerate(IMG_SPEC):
        with Image.open(IMG_DIR/f"{img_id}.png") as im:
            fig.paste(im, ((i%5)*256, (i//5)*286))
        d.text(((i%5)*256+8, (i//5)*286+260), img_id, fill="black")
    fig.save(OUT/"image_contact_sheet.png")

def judge(q, pred):
    """冻结的宽松语义判分；中英同义词可认，矛盾/多个不同计数不算正确。"""
    p = pred.lower().strip()
    if q["type"] == "exist":
        negative = bool(re.search(r"没有|不存在|没看到|未见|无[蓝红黄绿]|\bno\b|\bnot\b|\bnone\b|\bwithout\b", p))
        # 先识别否定，避免指南中 '有' in '没有' 的误判。
        if negative and re.search(r"\byes\b", p):
            return False
        if negative:
            return q["gold"] == "没有"
        positive = bool(re.search(r"有|存在|\byes\b|\bthere (?:is|are)\b", p))
        return positive and q["gold"] == "有"
    if q["type"] == "attr":
        square = bool(re.search(r"方形|正方|矩形|\bsquares?\b|\brectangles?\b", p))
        circle = bool(re.search(r"圆|\bcircles?\b|\bround\b", p))
        return (square and not circle) if q["gold"] == "方形" else (circle and not square)
    numbers = set(re.findall(r"(?<!\d)\d+(?!\d)", p))
    for pattern, n in [(r"零|\bzero\b", "0"), (r"一|\bone\b", "1"),
                       (r"二|两|\btwo\b", "2"), (r"三|\bthree\b", "3"),
                       (r"四|\bfour\b", "4"), (r"五|\bfive\b", "5")]:
        if re.search(pattern, p):
            numbers.add(n)
    return numbers == {q["gold"]}

def guide_judge(q, pred):
    """原指南子串规则仅作旁列核对，不把 '没有' 判为存在。"""
    p = pred.replace(" ", "")
    return ("没有" in p and "有" in p) if q["gold"] == "没有" else q["gold"] in p

def dpo_numbers(values):
    pw, rw, pl, rl, beta = values
    margin = beta * ((pw-rw)-(pl-rl))
    loss = max(-margin, 0) + math.log1p(math.exp(-abs(margin)))
    return dict(policy_chosen=pw, reference_chosen=rw, policy_rejected=pl,
                reference_rejected=rl, beta=beta, margin=margin, loss=loss)

def set_seed():
    import numpy as np
    import torch
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(4)

def load_model(model_path, train=False, adapter=None, quantized=False):
    import torch
    from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
    set_seed()
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    kwargs = dict(torch_dtype=torch.float32, device_map={"":"cuda:0"},
                  attn_implementation="eager", local_files_only=True)
    if train or quantized:
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True,
            bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float32)
    model = AutoModelForVision2Seq.from_pretrained(model_path, **kwargs)
    if train:
        from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, PeftModel
        # DPO 同时保留两条 policy 计算图；检查点只重算激活，不缩减训练步数或数据。
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=bool(adapter),
            gradient_checkpointing_kwargs={"use_reentrant": False})
        if adapter:
            model = PeftModel.from_pretrained(model, str(adapter), is_trainable=True)
        else:
            model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
                target_modules=["q_proj","k_proj","v_proj","o_proj"], task_type="CAUSAL_LM"))
        model.print_trainable_parameters()
        assert model.is_loaded_in_4bit
        assert all(not p.requires_grad for n,p in model.named_parameters() if "lora_" not in n)
    elif adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False)
    model.config.use_cache = False if train else True
    return processor, model

def prompt_text(processor, q):
    return processor.apply_chat_template([{"role":"user", "content":[
        {"type":"image"}, {"type":"text","text":q}]}], add_generation_prompt=True)

def image(img_id):
    from PIL import Image
    return Image.open(IMG_DIR/f"{img_id}.png").convert("RGB")

def encode(processor, img_id, q, response):
    """以实际处理后的 token 检查前缀，不能用原始字符串长度作掩码。"""
    import torch
    img = image(img_id)
    prompt = prompt_text(processor, q)
    pref = processor(text=[prompt], images=[img], return_tensors="pt")
    full = processor(text=[prompt+response+processor.tokenizer.eos_token], images=[img], return_tensors="pt")
    plen = pref["input_ids"].shape[1]
    assert torch.equal(pref["input_ids"], full["input_ids"][:,:plen]), "答案拼接改变了 prompt token 前缀"
    labels = full["input_ids"].clone()
    labels[:,:plen] = -100
    assert labels.ne(-100).sum()>0
    full["labels"] = labels
    return full, plen

def device_batch(batch, model):
    return batch.to(next(model.parameters()).device)

def seq_logprob(model, batch):
    import torch
    inputs = {k:v for k,v in batch.items() if k != "labels"}
    logits = model(**inputs).logits[:,:-1,:].float()
    target = batch["labels"][:,1:]
    mask = target.ne(-100)
    token_lp = torch.log_softmax(logits,dim=-1).gather(-1,target.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return token_lp[mask].sum()

def train(args):
    import torch
    from bitsandbytes.optim import PagedAdamW32bit
    out = OUT/args.run
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    adapter = OUT/"sft"/"adapter" if args.mode=="dpo" else None
    processor, model = load_model(args.model, train=True, adapter=adapter)
    trainable, total = model.get_nb_trainable_parameters()
    params = [p for p in model.parameters() if p.requires_grad]
    lr = 1e-4 if args.mode=="sft" else 5e-5
    optimizer = PagedAdamW32bit(params,lr=lr,weight_decay=0.01)
    steps = args.steps or (60 if args.mode=="sft" else 40)
    data = SFT_DATA if args.mode=="sft" else PRE_PAIRS + (EXTRA_PAIRS if args.pairs=="17" else [])
    batches = []
    for sample in data:
        if args.mode=="sft":
            b,_ = encode(processor,sample["img"],sample["prompt"],sample["answer"])
            batches.append(device_batch(b,model))
        else:
            bw,_ = encode(processor,sample["img"],sample["prompt"],sample["chosen"])
            bl,_ = encode(processor,sample["img"],sample["prompt"],sample["rejected"])
            batches.append((device_batch(bw,model),device_batch(bl,model)))
    # DPO 关闭 dropout，参考模型也关闭适配器；参考对数概率可缓存，因为基座始终冻结。
    references=[]
    model.eval() if args.mode=="dpo" else model.train()
    if args.mode=="dpo":
        with torch.no_grad(),model.disable_adapter():
            for bw,bl in batches:
                references.append((seq_logprob(model,bw).detach(),seq_logprob(model,bl).detach()))
        # 检查点在 train 模式才生效；显式关闭 Dropout，使 policy/reference 似然可比。
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0
        model.train()
    config=dict(mode=args.mode,run=args.run,steps=steps,seed=SEED,pairs=len(data),beta=args.beta,
        lr=lr,r=8,alpha=16,dropout=0.05 if args.mode=="sft" else 0,
        optimizer="PagedAdamW32bit",weight_decay=0.01,batch_size=1,
        model_path=args.model,reference="frozen NF4 base with adapter disabled",
        trainable=trainable,total=total,trainable_percent=100*trainable/total,
        starting_adapter=str(adapter) if adapter else "new LoRA",quantization="NF4 4-bit",
        sequence_logprob="sum over response tokens including EOS, prompt excluded",
        gradient_checkpointing=args.mode=="dpo",checkpoint_use_reentrant=False,
        attention_implementation="eager")
    write_json(out/"config.json",config)
    history=[]
    first_before=[p.detach().clone() for p in params]
    torch.cuda.reset_peak_memory_stats()
    loop_start=time.perf_counter()
    step=0
    with (out/"training.jsonl").open("x",encoding="utf-8",buffering=1) as log:
        epoch=0
        while step<steps:
            order=list(range(len(data)))
            random.Random(SEED+epoch).shuffle(order)
            for idx in order:
                if step==steps:
                    break
                optimizer.zero_grad(set_to_none=True)
                if args.mode=="sft":
                    loss=model(**batches[idx]).loss
                    row=dict(step=step+1,epoch=epoch+1,sample_index=idx,loss=float(loss.detach()))
                else:
                    bw,bl=batches[idx]
                    pw,pl=seq_logprob(model,bw),seq_logprob(model,bl)
                    rw,rl=references[idx]
                    margin=args.beta*((pw-rw)-(pl-rl))
                    loss=-torch.nn.functional.logsigmoid(margin)
                    row=dict(step=step+1,epoch=epoch+1,sample_index=idx,pair_id=data[idx]["id"],
                        loss=float(loss.detach()),margin=float(margin.detach()),
                        policy_chosen=float(pw.detach()),policy_rejected=float(pl.detach()),
                        reference_chosen=float(rw),reference_rejected=float(rl))
                assert torch.isfinite(loss)
                loss.backward()
                grad=torch.nn.utils.clip_grad_norm_(params,1.0)
                assert torch.isfinite(grad)
                optimizer.step()
                if step==0:
                    changed=sum(float((p.detach()-old).square().sum()) for p,old in zip(params,first_before))
                    assert changed>0,"首步适配器未更新"
                    write_json(out/"first_step_update.json",dict(squared_change=changed,grad_norm=float(grad)))
                    del first_before
                step+=1
                row.update(grad_norm=float(grad),elapsed_s=time.perf_counter()-loop_start)
                history.append(row)
                log.write(json.dumps(row,ensure_ascii=False)+"\n")
                print(json.dumps(row,ensure_ascii=False),flush=True)
            epoch+=1
    torch.cuda.synchronize()
    config.update(actual_steps=step,training_s=time.perf_counter()-loop_start,
        total_s=time.perf_counter()-started,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
        peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
    assert step==steps
    model.save_pretrained(out/"adapter")
    processor.save_pretrained(out/"processor")
    write_csv(out/"history.csv",history)
    write_json(out/"config.json",config)
    print("TRAINING_COMPLETE",args.run,config,flush=True)

def answer(processor,model,q):
    import torch
    model.eval()
    b=processor(text=[prompt_text(processor,q["q"])],images=[image(q["img"])],return_tensors="pt")
    b=device_batch(b,model)
    with torch.no_grad():
        generated=model.generate(**b,max_new_tokens=24,do_sample=False,use_cache=True)
    return processor.decode(generated[0,b["input_ids"].shape[1]:],skip_special_tokens=True).strip()

def evaluate(args):
    import torch
    quantized=args.eval_quantization=="nf4"
    out=OUT/("evaluations_nf4" if quantized else "evaluations")/args.run
    out.mkdir(parents=True,exist_ok=False)
    adapter=None if args.run=="baseline" else OUT/args.run/"adapter"
    start=time.perf_counter()
    processor,model=load_model(args.model,adapter=adapter,quantized=quantized)
    rows=[]
    for q in QUESTIONS+EXTRA_QUESTIONS:
        t=time.perf_counter()
        pred=answer(processor,model,q)
        row=dict(stage=args.run,**q,answer=pred,correct=judge(q,pred),
            guide_substring_correct=guide_judge(q,pred),inference_s=time.perf_counter()-t,
            question_set="builtin15" if q["id"].startswith("P") else "extra5")
        rows.append(row)
        write_json(out/"predictions.json",rows)
        print(json.dumps(row,ensure_ascii=False),flush=True)
    metrics=[]
    for name in ("builtin15","extra5"):
        for typ in ("count","attr","exist","all"):
            part=[r for r in rows if r["question_set"]==name and (typ=="all" or r["type"]==typ)]
            metrics.append(dict(stage=args.run,question_set=name,type=typ,n=len(part),
                correct=sum(r["correct"] for r in part),accuracy=sum(r["correct"] for r in part)/len(part),
                guide_correct=sum(r["guide_substring_correct"] for r in part)))
    write_csv(out/"predictions.csv",rows)
    write_csv(out/"metrics.csv",metrics)
    write_json(out/"runtime.json",dict(total_s=time.perf_counter()-start,
        evaluation_dtype="nf4_float32_compute" if quantized else "fp32",generation="greedy",max_new_tokens=24,
        quantized=quantized,note="重新加载同一基座再独立加载适配器，不叠加；NF4 为精度切换诊断，不替换原 FP32 指标"))
    print("EVALUATION_COMPLETE",args.run,metrics,flush=True)

def checks(args):
    import torch
    OUT.mkdir(parents=True,exist_ok=True)
    tests=[(QUESTIONS[10],"没有。",False),(QUESTIONS[11],"No.",True),
        (QUESTIONS[0],"There are two red circles.",True),
        (QUESTIONS[0],"12",False),(QUESTIONS[5],"a square",True),
        (QUESTIONS[5],"square and circle",False),(QUESTIONS[11],"No, yes.",False)]
    for q,pred,expected in tests:
        assert judge(q,pred)==expected,(q,pred)
    cases=[dpo_numbers(x) for x in [(1.2,0.8,0.9,0.6,0.1),
        (1.5,1.0,0.2,0.6,0.1),(0.8,1.2,0.9,0.6,0.1)]]
    for c in cases:
        got=-torch.nn.functional.logsigmoid(torch.tensor(c["margin"],dtype=torch.float64)).item()
        assert abs(got-c["loss"])<1e-12
    m=torch.tensor(0.01,requires_grad=True)
    (-torch.nn.functional.logsigmoid(m)).backward()
    assert m.grad.item()<0
    write_csv(OUT/"dpo_manual_checks.csv",cases)
    processor,model=load_model(args.model,train=True)
    b,plen=encode(processor,QUESTIONS[0]["img"],QUESTIONS[0]["q"],"2个。")
    b=device_batch(b,model)
    model.eval()
    with torch.no_grad():
        pw=seq_logprob(model,b)
        with model.disable_adapter():
            rw=seq_logprob(model,b)
    assert abs(float(pw-rw))<1e-4,"初始零 LoRA 应与参考基座一致"
    update_keys=[n for n,p in model.named_parameters() if p.requires_grad]
    from importlib.metadata import version
    env=dict(python=platform.python_version(),platform=platform.platform(),
        packages={k:version(k) for k in ["torch","torchvision","transformers","peft","accelerate",
                  "bitsandbytes","huggingface_hub","numpy","Pillow","matplotlib"]},
        gpu=torch.cuda.get_device_name(),visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        cuda=torch.version.cuda,trainable_parameter_tensors=len(update_keys),trainable_names=update_keys)
    write_json(OUT/"environment.json",env)
    write_json(OUT/"implementation_checks.json",dict(judge_tests=len(tests),dpo_cases=cases,
        dpo_margin_gradient=m.grad.item(),prompt_len=plen,answer_tokens=int(b["labels"].ne(-100).sum()),
        initial_policy_logprob=float(pw),initial_reference_logprob=float(rw),
        reference_requires_grad=rw.requires_grad,quantized_nf4=model.is_loaded_in_4bit,
        note="指南手算近似 0.693 不精确；margin=0.01 时正确 loss=0.68815968。"))
    print("IMPLEMENTATION_CHECKS_PASSED",env,flush=True)

def summarize():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    eval_dir=OUT/"evaluations"
    metrics=[]
    for p in sorted(eval_dir.glob("*/metrics.csv")):
        with p.open(encoding="utf-8-sig") as f:
            metrics.extend(list(csv.DictReader(f)))
    write_csv(OUT/"all_metrics.csv",metrics)
    stages=["baseline","sft","dpo12","dpo12_gc","dpo17","dpo17_gc","beta005","beta05"]
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for typ,ax in zip(["count","attr","exist"],axes):
        selected=sorted([r for r in metrics if r["question_set"]=="builtin15" and r["type"]==typ],
                        key=lambda r:stages.index(r["stage"]))
        ax.bar([r["stage"] for r in selected],[float(r["accuracy"])*100 for r in selected])
        ax.set(title=typ,ylabel="Accuracy (%)",ylim=(0,105))
        ax.tick_params(axis="x",rotation=35)
    fig.tight_layout();fig.savefig(OUT/"accuracy_comparison.png",dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    for run in stages[1:]:
        p=OUT/run/"history.csv"
        if not p.exists():
            continue
        with p.open(encoding="utf-8-sig") as f:
            hist=list(csv.DictReader(f))
        x=[int(r["step"]) for r in hist]
        ax=axes[0] if run=="sft" else axes[1]
        ax.plot(x,[float(r["loss"]) for r in hist],label=run)
        if run!="sft":
            axes[2].plot(x,[float(r["margin"]) for r in hist],label=run)
    for ax,title in zip(axes,["SFT loss","DPO loss","DPO margin"]):
        ax.set(title=title,xlabel="Optimizer step");ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(OUT/"training_curves.png",dpi=180);plt.close(fig)
    rows=[]
    for run in stages:
        p=eval_dir/run/"predictions.json"
        if p.exists():
            rows.extend(json.loads(p.read_text(encoding="utf-8")))
    # 九条人工评分材料只准备原始回答；不能把自动正确率或 AI 初评冒充人工评分。
    review=[]
    for row in rows:
        if row["stage"] in ("baseline","sft","dpo17","dpo17_gc") and row["id"] in ("P01","P06","P12"):
            review.append(dict(stage=row["stage"],id=row["id"],image=row["img"],question=row["q"],
                gold=row["gold"],answer=row["answer"],human_faithfulness_1to5="",
                ai_suggested_score="",review_status="待核验",notes=""))
    write_csv(OUT/"faithfulness_9_review.csv",review)
    write_csv(OUT/"failure_candidates.csv",[r for r in rows if not r["correct"]])
    print("SUMMARY_COMPLETE",metrics,flush=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--mode",choices=["make_data","check","infer","sft","dpo","eval","summarize"])
    ap.add_argument("--model",default=str(ROOT/"models"/"SmolVLM-256M-Instruct"))
    ap.add_argument("--run",default=None)
    ap.add_argument("--steps",type=int)
    ap.add_argument("--beta",type=float,default=0.1)
    ap.add_argument("--pairs",choices=["12","17"],default="17")
    ap.add_argument("--eval-quantization",choices=["fp32","nf4"],default="fp32")
    ap.add_argument("--dpo_check")
    args=ap.parse_args()
    if args.dpo_check:
        print(json.dumps(dpo_numbers(list(map(float,args.dpo_check.split(",")))),indent=2));return
    if args.mode=="make_data":make_data()
    elif args.mode=="check":checks(args)
    elif args.mode in ("sft","dpo"):
        args.run=args.run or args.mode;train(args)
    elif args.mode in ("infer","eval"):
        args.run=args.run or "baseline";evaluate(args)
    elif args.mode=="summarize":summarize()

if __name__=="__main__":main()

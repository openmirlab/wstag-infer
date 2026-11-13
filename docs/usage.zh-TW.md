# wstag-infer 使用指南

`wstag-infer` 用文字描述定位音檔中的聲音，例如找出「玻璃碎裂聲」在哪些時間出現。它使用作者的小型 WSTAG checkpoint，回傳每 40 毫秒一格的匹配分數；需要事件區間時，由你指定門檻。它不建立搜尋向量索引，也不包含訓練程式。

## 安裝

需要 Python 3.10 以上；目前完整模型推論已在 Python 3.12 的 CPU 和 CUDA 跑通。TorchAudio 是模型推論所需的選用依賴，請安裝 `model` extra，並留意它的二進位版本要與 Torch 相容。

```bash
uv pip install 'wstag-infer[model] @ git+https://github.com/openmirlab/wstag-infer.git'
```

也可以在這個 repo 內執行 `uv sync --extra model`。目前沒有 PyPI 發布版。權重不在 Git 或安裝套件裡；第一次 `load()` 會向作者的 Hugging Face 模型頁下載固定版本的設定、權重和詞表，並逐一檢查 SHA-256。

## 最簡單的用法

```python
from wstag_infer import WSTAGSession

with WSTAGSession(device="cpu") as model:
    result = model.infer(
        "recording.wav",
        ["the sound of glass breaking", "a siren wailing"],
        threshold=0.5,
    )

for event in result.events:
    print(event.phrase, event.start, event.end, event.peak_score)
```

把 `recording.wav` 換成自己的 WAV 或 FLAC 檔。`result.events` 是超過門檻後合併相鄰時間格得到的區間。只想看原始分數曲線時，省略 `threshold`：

```python
from wstag_infer import localize

result = localize("recording.wav", "the sound of glass breaking", device="cpu")
for frame in result.frames:
    print(frame.start, frame.end, frame.score)
```

`localize()` 每次呼叫都建立並關閉模型。大量音檔請重用 `WSTAGSession`，避免每段都重新載入權重。Session 提供 `load()`、`infer()`、`release()`、`close()`、`status` 和 `cache_info()`；同一個 Session 不應同時被多個執行緒呼叫 `infer()`。

命令列也能輸出 JSON：

```bash
wstag-infer recording.wav 'the sound of glass breaking' --threshold 0.5 --device cpu
```

## 結果怎麼看

| 欄位 | 意義 |
| --- | --- |
| `frames` | 每個描述的逐格分數，`start`／`end` 是原音檔的秒數，區間為 `[start,end)`。 |
| `events` | 呼叫者指定門檻後，連續通過時間格形成的近似事件區間。未給門檻時是空的。 |
| `windows` | 每次模型呼叫實際讀到與評分到的時間範圍。 |
| `analyzed_until` | 最後實際評分到的時間；尾端不足一格時可能略早於檔案結尾。 |
| `checkpoint_revision` | 本次使用的作者 checkpoint 固定版本。 |

`result.to_dict()` 可直接交給 `json.dumps()`。它也附上 `semantics`，說明時間格、重疊視窗與分數的定義。

**分數不是校準過的事件機率。** `0.5` 只是示例門檻，不保證適用於所有描述或音效；實際使用時應查看分數曲線並用自己的素材決定門檻。40 毫秒是輸出格距，不代表模型能把聲音的起訖精確判到 40 毫秒。模型只認固定英文詞表，未知單字會變成 `<unk>`；先試簡短、常見、英文小寫的聲音描述。

音訊會平均成 mono、重採樣到 32 kHz。超過 10 秒的音檔以 10 秒視窗、每 8 秒前進一次；同一時間格若被兩個視窗覆蓋，取較高分。短於 0.32 秒的輸入會先補零，再刪除超出原始長度的分數格。這些選擇與實際評分範圍都保留在結果中。

## 離線使用

在有網路的機器上準備作者的固定 snapshot，包含三個檔案：

```bash
hf download wsntxxn/cnn8rnn-w2vmean-audiocaps-grounding \
  config.json model.safetensors vocab.json \
  --revision a9eb20c3e2135e0ca82d747bb99ae10ee4eaec66 \
  --local-dir ./wstag-assets
```

把 `wstag-assets` 複製到離線機器後：

```python
from wstag_infer import WSTAGSession

with WSTAGSession(
    device="cpu",
    checkpoint_dir="./wstag-assets",
    local_files_only=True,
) as model:
    result = model.infer("recording.wav", "the sound of glass breaking")
```

套件會在載入前重新核對三個檔案的 SHA-256；缺檔或檔案不同會直接報錯。若使用預設 Hugging Face 快取，位置通常是 `~/.cache/huggingface/hub`；也可用 `cache_dir=` 指定另一個快取資料夾，並以 `cache_info()` 檢查是否已備妥。

## 與 FineLAP 的關係

兩者都能做「音檔＋文字描述 → 時間上的分數與事件區間」。WSTAG 的名目分數格距是 40 毫秒，FineLAP 是 160 毫秒；格距較密不表示聲音邊界一定比較準。兩個模型的分數尺度也不能直接比較。要選哪個，請用自己的音效類型與描述句檢查命中、漏報及邊界，而不要只看分數或格距。

## 來源與授權

原始研究與模型由 Xuenan Xu、Ziyang Ma、Mengyue Wu、Kai Yu 製作，見[論文](https://arxiv.org/abs/2401.02584)、[研究程式](https://github.com/wsntxxn/TextToAudioGrounding)與[作者模型頁](https://huggingface.co/wsntxxn/cnn8rnn-w2vmean-audiocaps-grounding)。這個套件只維護推論入口；OpenMIRLab 沒有訓練或重新發布權重。程式與模型的來源、固定版本和授權說明在[來源文件](source-provenance.md)。

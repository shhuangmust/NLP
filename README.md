---
tags: NLP, AI
---
# NLP(自然語言處理)預訓練模型
---
## 目錄
- [NLP簡介(基礎概念)](#基礎概念)
- [第一步驟：文字的表示方法](#文字(字詞)的表示方法)
- [第二步驟：建立語言模型](#語言模型)
- [基礎任務](#基礎任務)
- [應用任務](#應用任務)
- [文字的表示方法：Word2Vec實作](#Word2Vec實作)
- [神經網路語言模型(NNLM)](#神經網路語言模型)
- [自注意力機制(Self-Attention)](#Self-attention)
- [Transformer](#Transformer)
- [Bert](#Bert)

---

::: success
**甚麼是NLP(Natural Language Processing)自然語言處理**
:::

- 自然語言通常指的是人類語言
- NLP通常指的是書寫文字，若包括語音則稱之為HLP人類語言
- NLP指的就是，用電腦了解和生成自然語言的各種理論和方法
- 常見應用為：Google翻譯、聊天機器人、OCR，中文分詞，程式自動補全

## 基礎概念
:::success
**人工智慧三大階段**

**運算智慧->感知智慧->認知智慧**
:::

- 運算智慧：例如運算能力，1960年代時電腦早已超越人類
- 感知智慧：例如影像/語音辨識等機器學習感知能力，已達人類的水準
- 認知智慧：例如認知人類語言，和人類還有一大段差距

### 自然語言處理難度
- **抽象性**："車"可以表示各種交通工具——汽車、火車、自行車
- **組合性**：BIG5中文標準共收錄6,763個中文字，順序不同組合語義也不同，無法像下棋使用窮舉法解決
- **歧義性**："蘋果"可以指水果，也可以指一家公司或手機。兩個句子，如"曹雪芹寫了紅樓夢"和"紅樓夢的作者是曹雪芹"，形式不同但是語義是相同的
- **進化性**：新詞彙層出不窮，如“葛格”、“汪星人”。舊詞彙被指定新的含義，如“是在哈囉”、“杯具”
- **非規範性**：音近詞（“為什麼”→“為蛇麼”，“女兒”→“女鵝”），單字的簡寫或變形（please→pls、cool→coooooooool），新造詞”黑人問號”、”傻眼貓咪”、”突破盲腸”
- **主觀性**：打籃球”是一個詞還是兩個詞?無法制定標註標準，很難透過眾包的方式招募大量的標注人員
- **知識性**："張三打了李四，然後他倒了"，其中的“他”指的是誰？
- **難移植性**：不同的領域有不同的解決模型，難使用統一的技術或模型加以解決

### NLP的四大建設

![](https://i.imgur.com/vcEL3q3.png)

- **資源建設**：需要花費大量的人力和物力建構詞典（Dictionary或Thesaurus）、規則庫
- **基礎建設**：分詞(又稱斷詞)、詞性標注、句法分析和語義分析
- **應用任務**：作為產品直接被終端使用者使用，例如資訊取出、情感分析、問答系統、機器翻譯和對話系統等
- **應用系統**：在某一領域的綜合應用，例如智慧教育領域，可用文字分類、回歸實現試題智慧評閱

### NLP可完成的任務類別
- **回歸**：將輸入文字映射為一個連續的數值，例如幫文章打分數、案件刑期的預測
- **分類**：判斷一個輸入的文字所屬的類別，例如垃圾郵件辨識任務中，將一封郵件分為正常和垃圾兩類
，情感分析中，將使用者的情感分為褒義、貶義或中性三類
- **比對**：判斷兩個輸入文字之間的關係，例如論文比對
- **解析**：對文字中的詞語進行標注或辨識詞語之間的關係，中文分詞、命名實體辨識
- **生成**：特指根據輸入生成一段自然語言，如機器翻譯、文字摘要、圖型描述生成、自動詩詞產生

### NLP的研究歷史

![](https://i.imgur.com/YIdyZjV.png)

:::info
### 小規模專家知識時代(1970-1990)
語料規模以及運算能力的限制，只能做規則性的NLP處理，效果有很大的瓶頸
:::

:::success
### 大規模語料庫統計模型時代(1990-2010)
- 電腦運算速度和儲存容量的快速增加，統計方法的成熟，在自然語言處理領域得以大規模應用
- 機器翻譯和自動問答成功
- 利用經驗性規則將原始的自然語言輸入轉化為機器能夠處理的向量形式，稱為特徵提取或特徵工程
:::

:::warning
### 大規模語料庫深度學習時代(2010-2017)
- 深度神經網路為基礎的表示學習
- 語音辨識、電腦視覺十分成功
- Google翻譯在2017開始換成深度學習的模型
- 過度依賴於大規模有標注資料，人力成本過於高昂
:::

:::danger
### 預訓練模型(2018-至今)
- 預先訓練一個初始模型，需依賴大規模有標注資料
- 在下游任務（也稱目標任務）上，繼續對該模型進行微調，提高下游任務準確率
- 遷移學習（Transfer Learning）思想的一種應用
:::

### 預訓練模型

### 取之不盡，用不之竭的標注資料
- 圖書、網頁等文字資料規模近乎無限，非常容易獲得的預訓練資料
- 不需人工標注資料的預訓練學習方法，稱為自監督學習（Self-supervised Learning）

### 「詞向量」的新觀念
- 早期的一個詞代表的意思為"固定"的向量，稱之為靜態詞向量預訓練模型
- 後來的一個詞，可以由不同可能的向量組成(不同向量代表不同意義)，稱之為動態詞向量預訓練模型
- 2018年來，以BERT、GPT為代表的超大規模預訓練語言模型，達到或甚至超過了人類水準

### 巨型的預訓練詞向量模型
- Self-attention(自注意力)為基礎的Transformer模型顯著地提升了對於自然語言的建模能力
- OpenAI推出的GPT-3，是一個具有1,750億個參數
- 經過小樣本學習，即可完成文字生成任務

---

## 文字(字詞)的表示方法

![](https://i.imgur.com/cUdfS9f.png)

### 什麼是語言模型(Language Model)
- 一個語言模型函數p=f(x)，x是一個值，p是一個機率值
- p1=f(‘recognize speech’) = 0.9
- p2 =f(‘wreck a nice beach’)= 0.01
- p1>p2，所以這段語言資料表示的是p1的意思
- 我們的目的，就是建立起語言模型，讓我們可以針對文字進行計算、預測，以及應用
- 第一件事情，先找到文字(字詞)在電腦中如何表現

### 文字(字詞)在電腦中如何表現
- ASCII、UNICODE
- Halal、清真、حلال 都擁有一樣的語言意思，但是ASCII不同
- "A"、"B"的ASCII數學計算距離比"A"、"Z"靠近，但是語言實務上卻沒有意義
- 所以還要發展另外的模型，處理語言模型
  
![](https://i.imgur.com/Sk0B7BC.png)

### 文字(字詞)變成電腦可處理的值
- 某個word，通過一定的方法，映射或嵌入（embedding）到另一個數值向量空間，稱為embedding
- 為了電腦處理，要把文字原始的高維度進行降維，也就是高維空間映射到低維空間
- 例如：“apple on an apple tree”，將這些詞彙放置到一個高度維度空間["apple", "on", "an", "tree“]，輸出成低維度向量空間[1,1,1,1] (apple出現一次，on出現一次，an出現一次，tree出現一次)

### 字詞常見向量表示法
- 一個字詞，要如何用向量來表示他的意思
- **獨熱(碼) One-hot (encoding)**
- **詞袋模型 Bag of words**
- **分散式表示法**
    - **以頻率為基礎**：Count Vector、TF-IDF Vector、Co-Occurence Vector
    - **以預測為基礎(Word embedding)**
    - **基於矩陣的分佈表示**
    - **基於Clustering聚類的分佈表示**
    - **基於神經網絡的分佈表示**：CBOW、SKIP-GRAM

### 1.獨熱(碼)表示(One-hot)
![](https://i.imgur.com/yTVFLCx.png)

#### 獨熱(碼)的問題
- 全部都正交，沒有相似度的呈現，無法用COS找相似度
- 維度太高
- 機器學習時，有資料稀疏（Data Sparsity）問題
- “漂亮”，“美麗”，雖然它們之間很相似，系統無法對“美麗”加權
- 獨熱(one-hot)用來表示字詞，用獨熱把出現過的詞結合在一起，就是用讀熱碼表達句子的方式
#### [程式碼示範](https://github.com/joshhu/nlp_must2022/blob/main/2_3(one_hot).ipynb)

### 2.Bag of words(詞袋模型)
- 用來表達句子的方式
- 找出所有字詞
- 編輯成詞向量模型
- 統計每句話(或者每篇文章)的詞向量模型
- 計算兩句話(或者兩篇文章)是否相似，就是計算空間中的cosine距離
![](https://i.imgur.com/4Frzfl1.png)
![](https://i.imgur.com/fU2jmq5.png)
![](https://i.imgur.com/k6J0q9S.png)
![](https://i.imgur.com/0ukOffi.png)
![](https://i.imgur.com/MIcQQVb.png)


#### 詞袋表示法的問題
- 沒有考慮詞的順序資訊：例如“張三打李四”和“李四打張三”，包含的詞相同，詞序不同，詞袋表示結果一樣
- 無法融入上下文資訊。例如“不喜歡”，只能將"不"、"喜歡"兩個詞的向量相加
- 隨著文件的不斷增長，詞彙表的增長將會導致文件向量不斷的增長，表現為文件向量的維度不斷增加，造成更嚴重的資料稀疏問題
- 減小詞彙表大小就成為了主要的研究議題


### 3.詞的分散式表示法
- 一個詞並非本身所決定，是「分散在其旁」的其它字詞所決定
- JR Firth (1957:11)表示：一個詞的意思由其前後文決定(You shall know a word by the company it keeps”)
![](https://i.imgur.com/om17G4G.png)
- 例如："我"這個字如何表示 (假設"我"出現在下列三個句子當中)
![](https://i.imgur.com/q5O5UXI.png)
![](https://i.imgur.com/CbUYpmy.png)
- 經過統計後，「我」這個詞的向量就是
![](https://i.imgur.com/YMvrcLS.png)
- 改進方法：TF-IDF
    - Term Frequency–Inverse Document Frequency 
    - 字詞的重要性隨著它在檔案中出現的次數成正比 (TF- Term Frequency)
    - 隨著它在語料庫中出現的頻率成反比 (IDF-Inverse Document Frequency)

#### 分散式表示法問題
- 高頻詞誤導計算結果。例如：“我”、“的"
- 仍然存在稀疏性的問題
- 學術上的解決方法：TF-IDF，PMI，SVD
- TF-IDF
    - Term Frequency–Inverse Document Frequency 
    - 字詞的重要性隨著它在檔案中出現的次數成正比 (TF- Term Frequency)
    - 隨著它在語料庫中出現的頻率成反比 (IDF-Inverse Document Frequency)
- PMI (Pointwise Mutual Information)
    - 資訊理論中的點相互資訊
    - [範例](https://github.com/joshhu/nlp_must2022/blob/main/2_1(pmi).ipynb)
- SVD (Singular Value Decomposition) 
    - 奇異值分解
    - [範例](https://github.com/joshhu/nlp_must2022/blob/main/2_2(svd).ipynb)
    - 矩陣規模較大時，奇異值分解的執行速度非常慢
- Word-Embedding
    - 詞嵌入表示法

### 詞嵌入表示法
- 使用一個連續、低維、稠密的向量來表示詞，稱為詞嵌入表示法
- Distribution Representation 最早是在 1986 年 Hinton 的論文 – “ Learning distributed representations of concepts ” 中被提出
- 而 Word Embedding 則是 Distribution representation 的一種型態，可以克服 One-Hot representation 的缺點 。
- 將vector每個元素由整數改成浮點數
![](https://i.imgur.com/Yl1OZZW.png)
![](https://i.imgur.com/sHBHjLQ.png)

---

## 語言模型

- 學會了文字(字詞)的表示之後，接下來要處理一句話的語言模型
- 數個文字，會組合成有意義的字詞，例如："二"、"手"，組合成有意義的"二手"，"二手"、"機車"，會組成有意義的"二手機車"
    - "二"、"手機"、"車"????
- 英文為代表的印歐語系（Indo-European languages）中，詞之間通常用分隔符號（空格等）區分
- 中文為代表的漢藏語系（Sino-Tibetan languages），以阿拉伯語為代表的閃-含語系（Semito-Hamitic languages）中，不包含明顯的詞之間的分隔符號
    - 進行後續的自然語言處理，需要首先對不含分隔符號的語言進行分詞（Word Segmentation）

### N元語言模型
- 在指定詞序列𝑤_1 𝑤_2⋯𝑤_(𝑡−1)的條件下，對下一時刻𝑡可能出現的詞𝑤_𝑡的條件機率𝑃(𝑤_𝑡 |𝑤_1 𝑤_2⋯𝑤_(𝑡−1))進行估計
- 例如："我喜歡讀書"，我喜歡”，希望得到下一個詞為“讀書”的機率
    - 𝑃(讀書│我喜歡)
![](https://i.imgur.com/n9puPey.png)

### 馬可夫假設
- 每個字詞，會跟前n個字詞有關係，n可以為1,2,3,...
- N越大，考慮的歷史越完整，稱為N-gram
- N太大計算會太複雜，馬可夫假設當前的字僅與前幾個有限的字相關
- 實務上，n<3，稱為一元(unigram)、二元(bigram)、三元(trigram)
![](https://i.imgur.com/OIw6rHv.png)

### 貝氏機率計算n-gram
- N-gram
![](https://i.imgur.com/F8J1T2S.png)
- 1-gram
![](https://i.imgur.com/prq3Zfn.png)
- 2-gram (Bi-gram)
![](https://i.imgur.com/FLVsu2P.png)
![](https://i.imgur.com/CNw4Jco.png)
![](https://i.imgur.com/gGlSASe.png)
![](https://i.imgur.com/FHVbZBP.png)
- 3-gram或trigram：和前面二個詞有關
![](https://i.imgur.com/CpRWO5m.png)
![](https://i.imgur.com/vd0Djtm.png)

### N-gram的問題
- 英文單詞的界限是很明顯的，中文”詞”的劃分很複雜
- 浮點數下溢，也就是機率太小了，超出了浮點數所能表示的範圍
    - 解決辦法：在前面加一個LOG，讓它變成連加
- [練習](https://github.com/joshhu/nlp_must2022/blob/main/3_0(ngram).ipynb)

---

## 基礎任務

### 中文分詞

- 中文為代表的漢藏語系（Sino-Tibetan languages），以阿拉伯語為代表的閃-含語系（Semito-Hamitic languages）中，不包含明顯的詞之間的分隔符號

### 正向最大符合(FMM)
- 正向最大符合（Forward Maximum Matching，FMM）分詞演算法，從前向後，掃描句子中的字串，儘量找到詞典中較長的單字，作為分詞的結果
- 傾向於切分出較長的詞，容易導致錯誤
- 中文詞的定義也不明確，如「新竹縣」是「新竹 縣」還是「新竹縣」
- 詞並沒有收錄在詞典，無法切出來
- [範例](https://github.com/joshhu/nlp_must2022/blob/main/4_0(fmm).ipynb)

### 子詞切分
- 語言往往具有複雜的詞形變化，例如happy, happily, happier, happiest...等。
- 如果僅以天然的分隔符號進行切分，不但會造成一定的資料稀疏問題，還會導致由於詞表過大而降低處理速度
- 處理方法是根據語言學規則，引入詞形還原（Lemmatization）或詞幹提取（Stemming）
- 詞形還原指的是將變形的詞語轉為原形，如將“computing”還原為“compute
- 詞幹提取則是將字首、尾碼等去掉，保留詞幹（Stem），如“computing”的詞幹為“comput”

### 詞幹還原
- 以統計為基礎的位元組對編碼（Byte Pair Encoding，BPE）演算法
- BPE參考：https://zhuanlan.zhihu.com/p/424631681
- 還有WordPiece、Unigram Language Model（ULM）演算法
![](https://i.imgur.com/vpBe6ub.png)
![](https://i.imgur.com/p6V4xF2.png)
![](https://i.imgur.com/l81pYPZ.png)
![](https://i.imgur.com/mr9rgje.png)
![](https://i.imgur.com/IECSzux.png)
![](https://i.imgur.com/JRMb32X.png)
![](https://i.imgur.com/SxWAVAe.png)
![](https://i.imgur.com/ghakKxI.png)

### WordPiece和ULM
- WordPiece選擇能夠提升語言模型機率最大的相鄰子詞進行合併
- BPE和WordPiece演算法的詞表大小都是從小到大變化，屬於增量法。ULM則是減量法，即先初始化一個大詞表，根據評估準則不斷捨棄詞表中的子詞，直到滿足限定條件
- Google推出了SentencePiece開放原始碼工具套件，其中整合了BPE、ULM等子詞切分演算法

### 詞性標注
- 詞性是詞語在句子中扮演的語法角色
- 也被稱為詞類（Part-Of-Speech，POS），詞性標注（POS Tagging）
- 例如："他喜歡下象棋。"->他/PN 喜歡/VV 下/VV 象棋/NN 。/PU

### 句法分析
- 句法分析（Syntactic Parsing）的主要目標是指定一個句子，分析句子的句法成分資訊，從而有助更準確地了解句子的含義，並輔助下游自然語言處理任務
- 最終的目標是將詞序列表示的句子轉換成樹狀結構
![](https://i.imgur.com/NAVBkhE.png)

### 語義分析
- 透過離散的符號及結構顯性地表示語義
- 詞義消歧（Word Sense Disambiguation，WSD）：不同上下文，確定其具體含義
- 多詞一義：“馬鈴薯”和“洋芋”

---

## 應用任務

### 資訊擷取
- IE(Information Extraction）資訊擷取，是從非結構化的文字中自動提取結構化資訊的過程
- 例如從網頁中，找出某固定欄位的價格資訊
- 結構化的資訊方便電腦進行後續的處理
- 例如
    - **命名實體辨識（Named Entity Recognition，NER）**
    - **關係取出（Relation Extraction）**
    - **事件取出(Event Extraction)**
:::success
#### 命名實體辨識(NER)
- 命名實體辨識（Named Entity Recognition，NER），是在文字中取出每個提及的命名實體
- 例如：人名、地名、機構名、書名、電影名和藥物名
:::
:::info
#### 關係取出（Relation Extraction）
- 辨識NER實體之間的語義關係
- 例如：夫妻、子女、工作單位和地理空間上的位置

:::
:::warning


#### 事件取出(Event Extraction)
- 辨識人們感興趣的事件以及事件所相關的時間、地點和人物等關鍵元素
- 往往使用文字中提及的具體觸發詞（Trigger）定義
- 例如：
![](https://i.imgur.com/qwXZsHO.png)
![](https://i.imgur.com/TOOx4dQ.png)

:::

---

### 情感分析
- 文字的褒、貶、或中性
![](https://i.imgur.com/SIADkrW.png)
![](https://i.imgur.com/eapsIHd.png)

### 問答系統
- 接受自然語言形式描述的問題，並透過檢索、比對和推理等技術獲得答案
- **檢索式問答系統**：答案來自固定的文字語料庫或網際網路
- **知識庫問答系統**：透過查詢資料庫等結構化形式，回答相關基礎知識
- **常問問題集問答系統**：透過歷史累積的FAQ(常問問題集)進行檢索

### 機器翻譯
- 從一種自然語言（來源語言）到另外一種自然語言（目的語言）的自動翻譯
- 世界上存在約7,000種語言，其中，超過300種語言擁有100萬個以上的使用者
- **理性主義**：以規則為基礎的方法(早被棄用)
- **經驗主義**：以資料為基礎的方法，機器翻譯領域表現為以語料庫（翻譯實例庫）為基礎
- **深度神經網路學習**：用深度學習神經網路翻譯，所有翻譯規則都被編碼在神經網路的模型參數

### 對話系統
- 以自然語言為載體，使用者與電腦透過多輪互動的方式實現特定目標。
- 完成特定任務、獲取資訊或推薦、獲得情感撫慰和社交陪伴
- **任務型對話系統**：任務導向型的對話系統，用於垂直領域自動業務助理，明確的任務目標，完成機票預訂、天氣查詢
- **開放域對話系統（Open-Domain Dialogue）**：閒聊、情感陪護等為目標，因此也被稱為聊天系統或聊天機器人（Chatbot）
- 三個步驟：
    - **自然語言了解（Natural Language Understanding，NLU）**：分析使用者話語的語義
    - **對話管理（Dialogue Management，DM）**：包括對話狀態追蹤（Dialogue State Tracking，DST）和對話策略最佳化
    - **自然語言生成（Natural Language Generation，NLG）**：相對較簡單，透過範本即可實現

### NLP任務種類
:::success
**文字分類：情感分類、垃圾郵件、新聞標題分類**
- **文字比對（Text Matching）**：即判斷兩段輸入文字之間的符合關係
- **複述關係（Paraphrasing)**：判斷兩個表述不同的文字語義是否相同）
- **蘊含關係(Entailment)**：根據一個前提文字，推斷與假設文字之間的蘊含或矛盾關係）
:::
:::info
**結構預測問題**
- **序列標注（Sequence Labeling）**：為輸入文字序列中的每個詞標注對應的標籤
- **序列分割**：如中文分詞結構預測問題
:::
:::warning
**圖結構產生**

:::
:::danger
**序列到序列(Seq2seq)**
- 機器翻譯
- 語音到文字
:::

### 如何評價語言模型性能
- 以困惑度（Perplexity，PPL）為基礎的“內部評價”
- 困惑度越小，表示單字序列的機率越大，也表示模型能夠更進一步地解釋測試集中的資料
![](https://i.imgur.com/O81B8hO.png)
![](https://i.imgur.com/77znHgs.png)


### 語料庫介紹
- **維基百科下載**：[https://dumps.wikimedia.org/zhwiki/](https://dumps.wikimedia.org/zhwiki/)
- **Wikimedia**：[https://dumps.wikimedia.org/wikidatawiki/entities/](https://dumps.wikimedia.org/wikidatawiki/entities/)
- **Common Crawl**
- **Huggingface Dataset**
- **大規模的中文語料庫收集**：[https://github.com/brightmart/nlp\_chinese\_corpus](https://github.com/brightmart/nlp_chinese_corpus)
- [https://github.com/crownpku/Awesome-Chinese-NLP](https://github.com/crownpku/Awesome-Chinese-NLP)

---

## Word2Vec實作

### 文字向量表示
- 如何表達'bank'這個文字?
- 使用稠密向量表示
- Word2Vec就是用來計算文字的向量

![](https://i.imgur.com/Z0fxprk.png)
![](https://i.imgur.com/aFHuN30.png)
![](https://i.imgur.com/uVaKWwP.png)




### Word2vec步驟
- 找一個語料庫(Corpus)，擁有越多越好的正常句子 (幾千萬個句子)

- 定義出固定大小的字彚表 (十萬個字詞)

- 字彚表中的每一個字都對應一個向量 (用100個維度來表示一個字詞)

- 語料庫中的每一句，每一個位置t，都有**一個**中間字c，和**一組**包圍字o

- c、o出現的機率=c和o之間的向量相似度

- 不斷調整每一個字的向量來最大化這個機率
- 例如：(Windows size=2)

![](https://i.imgur.com/QUWhCtc.png)
![](https://i.imgur.com/0EwWwK6.png)
![](https://i.imgur.com/vcwR7wn.png)
![](https://i.imgur.com/ZpVZSPB.png)
- 我們的目標，就是最小化J(O)
![](https://i.imgur.com/w38Pzqy.png)
- 訓練模型方法
    - 一個代表所有向量的總參數
    - 向量大小為維度x總字彚大小x2
    - 每個字都有兩個向量
    - 使用梯度下降最佳化
![](https://i.imgur.com/4m44WXf.png)
- [範例](https://github.com/joshhu/nlp_must2022/blob/main/6_0(gensim).ipynb)

---

## 神經網路語言模型

- 如何辨識同音異意的一段聲音
- 例如："recognize speech" 與 "wreck a nice beach"
- 看哪一個機率P(句子)大，就輸出哪個句子
![](https://i.imgur.com/AoIrs28.png)
![](https://i.imgur.com/e1zVv2V.png)

### 傳統N-gram語言模型
- N-gram語言模型，就是一個字一個字計算整句的機率P(句子)
- Counting base language model
- 沒出現過的字句機率是零(無法產生訓練資料庫沒出現的句子)
![](https://i.imgur.com/7VsuHqy.png) 
![](https://i.imgur.com/SgDLwoj.png)

### 神經網路語言模型
- Neural network base language model
![](https://i.imgur.com/9HMSy3a.png)

### RNN神經語言模型
- RNN就是紀錄之前說過的"多句"話，來預測下一句話
![](https://i.imgur.com/MTcHU6g.png)
![](https://i.imgur.com/5xpPgmV.png)

---

## Self-attention
- RNN提出後，無法平行處理是最大的問題
- 因此發展生自注意力機制(2017)
- Self-attention可以平行處理，可取代RNN
- 保有RNN看過整個sequence的能力
- 如下圖，給定一個向量a1,a2,a3,a4，經過self-attention訓練後，會產生一個考慮所有向量的輸出b1,b2,b3,b4
![](https://i.imgur.com/h8GnREd.png)
- 可找出字詞與前後字詞的相關性
- Multi-head attention可以產生多維角度的字詞關係
![](https://i.imgur.com/sBq0YyV.png)
![](https://i.imgur.com/smhdyT0.png)
- self-attention最著名的參考資料：[Attention is all you need](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)

### Self-Attention詳細做法

- Self-attention的想法，是利用<Q,K,V>代表注意力機制，Query表示a1，Key表示要跟a1求重要性的a2, a3...a4，算出之間的關係值&alpha;，再根據Value的值，決定a2,a3...a4哪個比較重要，求出值b
![](https://i.imgur.com/w9kHYK2.png)
- a的位置加權演算法
![](https://i.imgur.com/FHHxOq6.png)
- attention最著名的應用，就是google的Bert

## Transformer
- 一種基於自Self-attention的Seq2Seq模型，近年在圖像描述、聊天機器人、語音辨識以及機器翻譯等各大領域大發異彩
- Seq2Seq(Sequence to Sequence)，就是輸入一個序列的值，產生另外一個序列的值，輸入跟產生的序列長度不一定要相同
    - 例如：輸入"機器學習"，產生"machine learning"
    - 輸入一段語音，產生中文字幕
- 下圖為Transformer的架構圖

![](https://i.imgur.com/y90fJoF.png)
- 如上圖，Transformer分為encoder與decoder (Seq2Seq的型態)

### Transformer範例：把英文翻譯成中文
![](https://i.imgur.com/z3lBTYv.png)

- 我們並沒有告訴模型，Europe就是歐洲，citizens就是公民，但是經過訓練後，圖形顯示在相對應的點受到特別多的注意力(顏色黃色)
- 這就是靠著self-attention機制的運作

### Transformer encoder
- encoder目的，把輸入向量進行編碼，轉換成隱含前後資訊的向量
- 舉例來說，輸入一個"機器學習"的語音，會編碼成四個考量前後因素的向量
![](https://i.imgur.com/4gn0nbo.png)
- encoder詳細作法：搭配Positional Encoding位置資訊，先做multi-head attention
- 將上面輸出與輸入(residual-殘差)進行add，然後再進行Layer Norm
- 結果丟入全連結網路Feed Forward
- 再做一次add & Norm
- 輸出
![](https://i.imgur.com/eRsYmmT.png)
- Bert就是Transformer的encoder

### Transformer decoder
- decoder的目的，把轉換好的隱含向量加入，依次解碼成我們要的目的
- 舉例來說，由start開始，搭配encoder好的向量，依次推導出"機器學習"四個字
![](https://i.imgur.com/gXjdzvA.png)
- decoder詳細做法，基本上與encoder類似，只是多了一組參考encoder的self-attention，與改成Masked self-attention
- Masked self-attention就是只能往前參考資訊，不能像之前self-attention一樣參考往後的資訊
![](https://i.imgur.com/AItcqkZ.png)
- 關鍵點在於多出來的一組參考encoder的self-attention，稱之為cross-attention，參考encoder產生出來的&alpha;與v值，與decoder的q一起產生結果
![](https://i.imgur.com/ol8ZE5A.png)
![](https://i.imgur.com/LD4CXoO.png)

---

# Bert

- [Bert](https://arxiv.org/abs/1810.04805) (Bidirectional Encoder Representations from Transformers），用於自然語言處理（NLP）的預訓練技術，由Google在2018年提出
- 芝麻街角色Bert的名字來命名
- 採用自監督訓練法(Self-supervised learning)
- 兩種Bert的預訓練模型
    - BERT BASE模型，12層，768維，12個self attention head，110M參數
    - BERT LARGE模型，24層，1024維，16個self attention head，340M參數
- 訓練語料庫為BooksCorpus以及英語維基百科，單詞量分別是8億以及25億
- BERT在以下自然語言理解任務上的效能表現得最為卓越
    - GLUE（General Language Understanding Evaluation，通用語言理解評估）任務集（包括9個任務）
    - SQuAD（Stanford Question Answering Dataset，史丹佛問答資料集）v1.1和v2.0
    - SWAG（Situations With Adversarial Generation，對抗生成的情境）
![](https://i.imgur.com/vyvm6GV.jpg)

## Bert模型如何訓練而來

- 第一種訓練方式：Mask token prediction (遮住一個字，想辦法猜出來)
![](https://i.imgur.com/WY2yS7W.png)

- 第二種訓練方式：Next sentence prediction (猜兩個句子是否有意義的前後句子)
![](https://i.imgur.com/nXxKYQf.png)


## 如何使用Bert

- Bert為pre-trained model
![](https://i.imgur.com/F2wnzZX.png)


### Case 1
- 輸入一個句子，判斷其類別(例如正評/負評)
![](https://i.imgur.com/IR1whjP.png)

### Case 2
- 輸入一句話，輸出字的詞性(POS tagging)
- 不一定要詞性，同樣長度的預測應用都可以
![](https://i.imgur.com/WB4dQyg.png)

### Case 3
- 輸入兩句話，判斷是否矛盾
![](https://i.imgur.com/87QHFz0.png)

### Case 4
- 輸入一段文章，找出答案的起始位置與結束位置

![](https://i.imgur.com/TVeasTz.png)
![](https://i.imgur.com/D8A9XAZ.png)


### 範例：食品廣告分析
- [Hugging Face](https://huggingface.co/)：儲存全世界AI模型的儲存庫
    - bert的預訓練模型：HFL(哈爾濱大學提供的中文預訓練模型)
![](https://i.imgur.com/vk3oEKq.png)

- 範例資料庫：[Bert.ipynb](https://github.com/shhuangmust/AI/blob/f6e8d2b51c42243216a7b585adae884e6e018093/17.Bert.ipynb)
- 食品廣告是否違反衛生保健法範例資料
![](https://i.imgur.com/z0G4ZEX.png)

![](https://i.imgur.com/TMTFl3S.png)







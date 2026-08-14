# RSNA Knee Abnormality Detection: Report Labeling System

## 1. Objective
Extract high-precision, calibrated weak supervision labels from multilingual radiology reports (`train.csv`) to train image-only MRI models.

---

## 2. Extraction Architecture

The clinical NLP pipeline operates across four semantic states:
- **`positive`**: Explicit confirmation of pathology (probability = 0.95, loss_mask = True)
- **`negative`**: Explicit confirmation of normalcy / absence (probability = 0.05, loss_mask = True)
- **`uncertain`**: Equivocal, suspected, or low-confidence mentions (probability = 0.50, loss_mask = False)
- **`not_mentioned`**: Omitted structure (probability = 0.10, loss_mask = False)

---

## 3. Multilingual Ontological Coverage
- **English**: Comprehensive orthopaedic and musculoskeletal radiology lexicons.
- **German**: VKB-Ruptur, Innenbandläsion, Außenmeniskusläsion, Gonarthrose, Gelenkerguss.
- **Spanish**: Rotura del LCA, menisco interno/externo, artrosis femoropatelar, derrame articular.
- **French**: Rupture LCA, ménisque médial, gonarthrose, épanchement intra-articulaire.

---

## 4. Context & Negation Handling
- Scans up to 80-character context window preceding positive terms.
- Detects clause conjunctions (`ni`, `nor`, `sans`, `sin signos de`, `no evidence of`).
- Prioritizes **Findings** and **Impression** sections over clinical indications/history.

# 🎓 AI-Based Student Performance Analysis System

> **Developed by Mohamed Faisal** · Department of Artificial Intelligence and Data Science

A beginner-friendly Data Science and Machine Learning project that analyses student marks, attendance and study habits, **predicts each student's performance category**, and shows everything in a modern **Streamlit dashboard**.

---

## 1. What does this project do? (in simple English)

Imagine a teacher who has a spreadsheet with every student's attendance, internal marks, assignment marks, last semester's marks and daily study hours. This project:

1. **Cleans** the spreadsheet (fixes missing numbers, impossible values and duplicates).
2. **Analyses** it (averages, top and low performers, which factors are linked to good marks).
3. **Teaches a computer** (Machine Learning) to predict whether a student will be *Excellent, Good, Average* or *Needs Improvement*.
4. **Shows** the results with charts and a professional dashboard, plus personal improvement tips.

### Performance categories (based on final marks)

| Category | Final marks |
|---|---|
| Excellent | 80 and above |
| Good | 65 to 79.9 |
| Average | 55 to 64.9 |
| Needs Improvement | below 55 |

---

## 2. Features

| Requirement | Where it is in the project |
|---|---|
| Student data (name, ID, attendance, internal, assignment, previous sem, study hours) | `data/students.csv` |
| Handle missing and inconsistent data | `src/data_processing.py` → `clean_data()` |
| Statistical summaries, correlations, insights | `summary_statistics()`, `generate_insights()` |
| High and low performers | Dashboard → *High performers* / *Low performers* tabs |
| ML prediction + categories + accuracy | `src/model.py` (Random Forest) |
| Factors that influence performance | Feature-importance chart (Model Performance page) |
| Bar, pie, line, heatmap, scatter charts | `src/visuals.py` |
| Dashboard with filters | `app.py` |
| Login, About page, creator credit | `app.py` |

---

## 3. Folder structure

```
student-performance-ai/
├── app.py                  ← the Streamlit dashboard (run this)
├── train_model.py          ← runs the whole analysis in the terminal, saves charts
├── requirements.txt        ← list of Python packages to install
├── README.md               ← this guide
├── .gitignore
├── .streamlit/
│   └── config.toml         ← dashboard colour theme
├── data/
│   ├── generate_data.py    ← creates the sample CSV
│   └── students.csv        ← sample (synthetic) student data
├── src/
│   ├── __init__.py         ← empty file (makes src a Python package)
│   ├── config.py           ← settings: column names, colours, branding
│   ├── data_processing.py  ← cleaning + analysis (Pandas, NumPy)
│   ├── model.py            ← Machine Learning (Scikit-learn)
│   └── visuals.py          ← all charts (Matplotlib, Seaborn)
├── models/                 ← saved model (created by train_model.py)
└── outputs/                ← chart images (created by train_model.py)
```

---

## 4. Installation, step by step (VS Code)

### Step 1 – Install the tools (one time)
1. **Python 3.9 or newer**: download from <https://www.python.org/downloads/>. On Windows, tick **“Add Python to PATH”** in the installer.
2. **VS Code**: download from <https://code.visualstudio.com/>.
3. In VS Code open *Extensions* (`Ctrl+Shift+X`), search **Python** (by Microsoft) and click **Install**.

### Step 2 – Open the project
1. Unzip the project so you have a folder called `student-performance-ai`.
2. In VS Code: **File → Open Folder…** → choose `student-performance-ai`.
3. Open the terminal: **Terminal → New Terminal** (or press `` Ctrl+` ``). Make sure the path shown ends with `student-performance-ai`.

### Step 3 – Create a virtual environment (a private box for this project's packages)

**Windows**
```bash
python -m venv .venv
.venv\Scripts\activate
```
> If PowerShell says “running scripts is disabled”, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, press `Y`, then try again. Or use **Command Prompt** instead.

**Mac / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

You will see `(.venv)` at the start of the terminal line. 
VS Code may ask *“We noticed a new environment”* → click **Yes**. Otherwise press `Ctrl+Shift+P` → **Python: Select Interpreter** → choose the one inside `.venv`.

### Step 4 – Install the packages
```bash
pip install -r requirements.txt
```
This takes 1 to 3 minutes.

---

## 5. Run the project in three stages

### Stage 1 – Create the sample data (optional, the file is already included)
```bash
python data/generate_data.py
```
Creates `data/students.csv` with 150 students (plus a few intentional problems: missing values, impossible values, duplicates, messy names, so the cleaning step has real work to do).

### Stage 2 – Run the analysis and ML in the terminal
```bash
python train_model.py
```
You will see the cleaning report, statistics, insights, model accuracy and feature importance. It also saves the trained model to `models/` and all charts to `outputs/`. 
*Do this stage first: if it works, the dashboard will work.*

### Stage 3 – Start the dashboard
```bash
streamlit run app.py
```
Your browser opens at **http://localhost:8501**.

**Demo login:** username `admin` · password `admin123` (change these in `src/config.py`).

To stop the dashboard press `Ctrl+C` in the terminal.

---

## 6. Dashboard pages

| Page | What you can do |
|---|---|
| 📊 **Dashboard** | KPI cards (total students, average marks, attendance, study hours, model accuracy), bar chart, pie chart, line chart, heatmap, attendance-vs-marks scatter, AI predictions table. **Sidebar filters:** category, attendance range, individual students |
| 👤 **Student Profile** | Pick one student: marks, progress line, comparison with class average, AI prediction and personal tips |
| 🔮 **Predict Performance** | Move sliders for a new student and get an instant prediction with confidence |
| 🧪 **Model Performance** | Accuracy, precision, recall, F1, R², confusion matrix, feature importance, model comparison |
| 🗂️ **Data & Cleaning** | Raw vs cleaned data, cleaning report, summary statistics, CSV download |
| ℹ️ **About** | Developer and project details |

The creator credit appears in the **header** (“Developed by Mohamed Faisal”), the **footer** (“© 2026 Mohamed Faisal | AI & Data Science Student”), the **login page**, the **sidebar** and the **About page**.

---

## 7. How the code fits together

```
students.csv ─► data_processing.clean_data() ─► model.train_models() ─► app.py (dashboard)
                        │                              │                      ▲
                        └───────────── visuals.py ─────┴──────────────────────┘
```

| File | Job |
|---|---|
| `src/config.py` | One place for column names, colours, thresholds and the creator's name. Every other file imports from here, so they always agree. |
| `src/data_processing.py` | Cleans data, makes summaries and plain-English insights, gives rule-based tips. |
| `src/model.py` | Splits data, trains the Random Forest models, measures accuracy, makes predictions. |
| `src/visuals.py` | Every chart. Each function returns a figure that `app.py` displays. |
| `app.py` | Login, sidebar, pages, styling. |
| `train_model.py` | The same pipeline from the terminal. |

### How the data is cleaned
1. Names and IDs are tidied (extra spaces removed, `IRFAN KUMAR` becomes `Irfan Kumar`).
2. Duplicate student IDs are removed.
3. Impossible values (attendance above 100%, negative study hours, marks above 100) are treated as missing.
4. Rows with no final marks are dropped (nothing to learn from).
5. Remaining missing numbers are filled with the column **median** (the middle value, which is not distorted by extreme values).

---

## 8. The Machine Learning algorithm: Random Forest

**Inputs (features):** attendance, internal marks, assignment marks, previous semester marks, study hours per day.
**Outputs:** the performance **category** (classification) and the **final marks** (regression).

**Idea in simple words**
- A *decision tree* is a flow-chart of yes/no questions (“Is internal mark above 60? Is attendance above 75%?”) that ends in a prediction.
- One tree can make silly mistakes. A **Random Forest builds 200 trees**, each trained on a slightly different random sample of students, and lets them **vote**. The crowd is more reliable than any single tree.

**Why Random Forest for this project?** It works well on small tables, needs little tuning, handles different scales without extra work, and reports **feature importance**, which answers “what affects performance the most?”.

### How the model is tested
1. **Train/test split:** 80% of students (120) are used to learn, 20% (30) are hidden and used only for testing.
2. **5-fold cross-validation:** the data is split into 5 parts and the model is tested 5 times, each time on a different part. This is a steadier estimate than one split.
3. **Per-student predictions** in the dashboard come from cross-validation, so each student is predicted by a model that never saw that student.
4. The final model used for the *Predict* page is trained on all students.

### Metrics explained
| Metric | Meaning |
|---|---|
| **Accuracy** | Out of 100 students, how many got the correct category |
| **Precision** | When the model says “Excellent”, how often is it right? |
| **Recall** | Of all truly Excellent students, how many did it find? |
| **F1-score** | One number that balances precision and recall |
| **Confusion matrix** | Table of actual vs predicted categories (diagonal = correct) |
| **R²** | How much of the variation in marks the model explains (1.0 = perfect) |
| **MAE** | Average error of the predicted marks, in marks |

### Results on the included sample data
| Metric | Value |
|---|---|
| Test accuracy (30 unseen students) | 90.0% |
| 5-fold cross-validated accuracy | 71.3% (± 6.9%) |
| Weighted precision / recall / F1 | 90.1% / 90.0% / 89.9% |
| R² for predicted marks | 0.80 |
| Mean absolute error | 4.15 marks |

**Please read this honestly in your viva:** the test set has only 30 students, so 90% is partly luck; the cross-validated 71% is the more reliable figure. Neighbouring categories (for example 64 vs 66 marks) are naturally hard to separate. On this small dataset a simple Logistic Regression scores about the same as the Random Forest (see the *Model comparison* table), which is a normal finding and a good discussion point.

Feature importance on the sample data: Internal Marks (28%) > Assignment Marks (20%) > Previous Sem Marks (19%) > Study Hours (17%) > Attendance (15%).

### Sample charts (saved by `train_model.py` in `outputs/`)

| | |
|---|---|
| ![bar](outputs/01_bar_top_students.png) | ![pie](outputs/02_pie_categories.png) |
| ![line](outputs/03_line_progress.png) | ![heatmap](outputs/04_correlation_heatmap.png) |
| ![scatter](outputs/05_attendance_vs_marks.png) | ![importance](outputs/07_feature_importance.png) |

---

## 9. Use your own data

Upload a CSV in the sidebar (**📂 Use your own data**), or replace `data/students.csv`. It needs these exact column names:

```
Student_ID, Name, Attendance, Internal_Marks, Assignment_Marks, Previous_Sem_Marks, Study_Hours, Final_Marks
```
Attendance and marks are 0–100; study hours are per day. At least 30 students are needed.

> The included data is **synthetic** (made up by `generate_data.py`) and is for demonstration only. Do not present it as real student data.

---

## 10. Common problems

| Problem | Fix |
|---|---|
| `'python' is not recognized` | Reinstall Python and tick *Add to PATH*, or try `py` instead of `python` (Windows) |
| `ModuleNotFoundError: No module named 'streamlit'` | The virtual environment is not active. Run the activate command from Step 3, then `pip install -r requirements.txt` |
| `ModuleNotFoundError: No module named 'src'` | Run commands from the **project root folder** (the folder that contains `app.py`) |
| `data/students.csv was not found` | Run `python data/generate_data.py` |
| Browser does not open | Open <http://localhost:8501> manually |
| Port already in use | `streamlit run app.py --server.port 8502` |

---

## 11. Future improvements (ideas for a final-year project)

**Better machine learning**
- Hyperparameter tuning (`GridSearchCV`), and compare XGBoost / LightGBM, SVM and a small neural network.
- Explainable AI with **SHAP** or **LIME**: “why was *this* student predicted Average?”.
- Early-warning system: predict **at-risk** students *before* the final exam and send alerts.
- Time-series model using weekly attendance and test scores across the semester.
- Handle class imbalance (SMOTE) and add fairness checks across groups.

**Real data and deployment**
- Connect to a real database (SQLite / PostgreSQL / MySQL) instead of CSV.
- Role-based login (admin, teacher, student) with a proper authentication library and hashed passwords.
- Deploy free on **Streamlit Community Cloud** and add the live link to your GitHub README and LinkedIn.
- Containerise with Docker and add a FastAPI prediction API.

**More features**
- Export a PDF report per student; email or WhatsApp alerts for low attendance.
- Subject-wise analysis, semester comparison, clustering of similar students (K-Means).
- Recommendation engine for study plans; multi-language interface.
- Unit tests (`pytest`) and GitHub Actions for automatic checks.

---

## 12. Put it on GitHub (portfolio)

```bash
git init
git add .
git commit -m "AI-Based Student Performance Analysis System"
git branch -M main
git remote add origin https://github.com/<your-username>/student-performance-ai.git
git push -u origin main
```
Tip: run the dashboard, take a screenshot, save it as `docs/dashboard.png` and add `![Dashboard](docs/dashboard.png)` near the top of this README.

---

## 13. Technologies

Python · Pandas · NumPy · Matplotlib · Seaborn · Scikit-learn · Streamlit

---

**© 2026 Mohamed Faisal | AI & Data Science Student**

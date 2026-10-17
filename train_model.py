"""Run the whole analysis in the terminal (no dashboard needed).

    python train_model.py

It will:
  1. load and clean data/students.csv
  2. train and evaluate the Machine Learning models
  3. save the trained models to  models/student_models.joblib
  4. save every chart as a PNG into outputs/  (handy for screenshots / GitHub)
"""
import joblib
import matplotlib.pyplot as plt
import pandas as pd

from src import config as cfg
from src import visuals as vz
from src.data_processing import clean_data, generate_insights, summary_statistics
from src.model import train_models


def save(fig, name):
    path = cfg.OUTPUT_DIR / name
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.relative_to(cfg.BASE_DIR)}")


def main():
    cfg.OUTPUT_DIR.mkdir(exist_ok=True)
    cfg.MODEL_DIR.mkdir(exist_ok=True)

    print("\n=== STEP 1: Load and clean the data ===")
    raw = pd.read_csv(cfg.DATA_PATH)
    clean, report = clean_data(raw)
    print(f"Rows before cleaning : {report['rows_before']}")
    print(f"Duplicates removed   : {report['duplicates_removed']}")
    print(f"Missing values found : {report['missing_found']}")
    print(f"Impossible values    : {report['invalid_found']}")
    print(f"Rows after cleaning  : {report['rows_after']}")

    print("\n=== STEP 2: Summary statistics ===")
    print(summary_statistics(clean).to_string())

    print("\n=== STEP 3: Key insights ===")
    for note in generate_insights(clean):
        print(" -", note.replace("**", ""))

    print("\n=== STEP 4: Train and evaluate the models ===")
    bundle = train_models(clean)
    m = bundle["metrics"]
    print(f"Training students : {m['train_size']}   Test students: {m['test_size']}")
    print(f"Accuracy          : {m['accuracy']:.1%}")
    print(f"Precision (wtd)   : {m['precision']:.1%}")
    print(f"Recall (wtd)      : {m['recall']:.1%}")
    print(f"F1-score (wtd)    : {m['f1']:.1%}")
    print(f"5-fold CV accuracy: {m['cv_accuracy']:.1%} (+/- {m['cv_std']:.1%})")
    print(f"Marks prediction  : R2 = {m['r2']:.2f}, MAE = {m['mae']:.2f} marks, RMSE = {m['rmse']:.2f}")
    print("\nModel comparison (cross-validated accuracy):")
    print(bundle["comparison"].to_string(index=False))
    print("\nFeature importance:")
    for feature, value in bundle["importance"].items():
        print(f"  {cfg.FEATURE_LABELS[feature]:<22} {value:.1%}")

    print("\n=== STEP 5: Save model and charts ===")
    joblib.dump(
        {"classifier": bundle["classifier"], "regressor": bundle["regressor"], "features": cfg.FEATURES},
        cfg.MODEL_DIR / "student_models.joblib",
    )
    print("  saved models/student_models.joblib")

    scored = bundle["scored"]
    save(vz.bar_student_marks(scored, 10, top=True), "01_bar_top_students.png")
    save(vz.pie_categories(scored), "02_pie_categories.png")
    save(vz.line_class_progress(scored), "03_line_progress.png")
    save(vz.heatmap_correlation(scored), "04_correlation_heatmap.png")
    save(vz.scatter_vs_marks(scored, "Attendance", "Attendance (%)", "Attendance vs final marks"),
         "05_attendance_vs_marks.png")
    save(vz.scatter_vs_marks(scored, "Study_Hours", "Study hours per day", "Study hours vs final marks"),
         "06_study_vs_marks.png")
    save(vz.feature_importance_chart(bundle["importance"]), "07_feature_importance.png")
    save(vz.confusion_matrix_chart(bundle["confusion_matrix"], bundle["labels"]), "08_confusion_matrix.png")

    print("\nDone! Now start the dashboard with:  streamlit run app.py")


if __name__ == "__main__":
    main()

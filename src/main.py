import app, clean, data_preprocessing, eda, feature_selection, heatmap, label_analysis, merge_dataset, model_evaluation, predict, train_all_models

merge_dataset.merge_datasets()
clean.clean_dataset()
label_analysis.analyze_labels()
eda.plot_attack_distribution()
heatmap.generate_heatmap()
feature_selection.feature_selection()
data_preprocessing.prepare_dataset()
train_all_models.main()
model_evaluation.main()
predict.main()
app.main()
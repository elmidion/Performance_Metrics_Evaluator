import streamlit as st
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, confusion_matrix
import io
import os
from datetime import datetime
import pingouin as pg
from statsmodels.stats.proportion import proportion_confint
import numpy as np

def Excel_Generator(df, cross_tables, file_1_name, file_2_name):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Summary')
        for column, (conf_matrix, results, labels) in cross_tables.items():
            # Create a unique sheet name for each column (limited to 31 characters)
            sheet_name = f"Sheet_{common_columns.index(column) + 1}"

            # Write data to the sheet
            worksheet = writer.book.create_sheet(sheet_name)
            worksheet.cell(row=1, column=1).value = f"Cross Table for {column}"  # Add title to sheet
            
            # Add headers indicating the file sources for rows and columns
            worksheet.cell(row=2, column=2).value = f"Columns: {file_2_name}"
            worksheet.cell(row=3, column=1).value = f"Rows: {file_1_name}"

            # Write the confusion matrix labels and values
            worksheet.cell(row=4, column=1).value = ""  # Empty cell for alignment
            for idx, label in enumerate(labels, start=2):  # Adjust start index
                worksheet.cell(row=4, column=idx).value = label  # Column labels

            for idx, label in enumerate(labels, start=5):  # Adjust start index for row labels
                worksheet.cell(row=idx, column=1).value = label  # Row labels
            
            for r_idx, row in enumerate(conf_matrix, start=5):  # Adjust start row by +1
                for c_idx, value in enumerate(row, start=2):
                    worksheet.cell(row=r_idx, column=c_idx).value = value

            # Write results (TP, FP, FN, TN, Precision, Recall, F1)
            results_df = pd.DataFrame(results)
            start_row = len(conf_matrix) + 7  # Leave some space after the confusion matrix
            for idx, label in enumerate(results_df.index, start=start_row):
                worksheet.cell(row=idx, column=1).value = label  # Row labels
            for idx, label in enumerate(results_df.columns, start=2):
                worksheet.cell(row=start_row - 1, column=idx).value = label  # Column labels

            for r_idx, row in enumerate(results_df.values, start=start_row):
                for c_idx, value in enumerate(row, start=2):
                    worksheet.cell(row=r_idx, column=c_idx).value = value

    output.seek(0)
    processed_data = output.getvalue()
    return processed_data


def infer_column_type(data):
    if pd.api.types.is_numeric_dtype(data):
        return 'continuous'
    elif pd.api.types.is_categorical_dtype(data) or data.dtype == object:
        return 'categorical'
    else:
        return 'unknown'

def validate_and_transform_column(true_data, pred_data, is_categorical):
    if is_categorical:
        true_data = true_data.astype(str)
        pred_data = pred_data.astype(str)
        all_unique_values = pd.unique(pd.concat([true_data, pred_data], ignore_index=True))
        cat_mapping = {val: idx for idx, val in enumerate(all_unique_values)}
        true_data_encoded = true_data.map(cat_mapping)
        pred_data_encoded = pred_data.map(cat_mapping)
        true_data_encoded = true_data_encoded[true_data_encoded != -1]
        pred_data_encoded = pred_data_encoded[pred_data_encoded != -1]
        return true_data_encoded, pred_data_encoded, 'categorical', cat_mapping
    else:
        return true_data.astype(float), pred_data.astype(float), 'continuous', None

def calculate_icc(true_data, pred_data):
    icc_data = pd.DataFrame({
        'Measurements': pd.concat([true_data, pred_data], ignore_index=True),
        'ID': list(range(len(true_data))) + list(range(len(pred_data))),
        'Rater': ['True'] * len(true_data) + ['Pred'] * len(pred_data)
    })
    icc_result = pg.intraclass_corr(data=icc_data, targets='Measurements', raters='Rater', ratings='ID', nan_policy='omit').round(3)
    icc_value = icc_result.at[0, 'ICC']
    return icc_value

def calculate_metrics(true_data, pred_data, labels, cat_mapping=None):
    conf_matrix = confusion_matrix(true_data, pred_data, labels=labels)
    accuracy = accuracy_score(true_data, pred_data)
    precision = precision_score(true_data, pred_data, average='macro', zero_division=0)
    recall = recall_score(true_data, pred_data, average='macro', zero_division=0)
    f1 = f1_score(true_data, pred_data, average='macro', zero_division=0)
    results = {}
    for idx, label in enumerate(labels):
        TP = FP = FN = TN = 0
        for i in range(len(true_data)):
            true = true_data[i]
            pred = pred_data[i]
            if pred == label and true == label:
                TP += 1
            elif pred == label and true != label:
                FP += 1
            elif true == label and pred != label:
                FN += 1
            if true != label and pred != label:
                TN += 1
        results[label] = {'TP': TP, 'FP': FP, 'FN': FN, 'TN': TN}

    n_samples = len(true_data)
    metrics = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}
    ci_results = {}
    for metric_name, metric_value in metrics.items():
        if metric_value == 1.0:
            ci_results[metric_name] = f"{metric_value:.3f}±0.000"
        else:
            ci = proportion_confint(int(metric_value * n_samples), n_samples, alpha=0.05, method='wilson')
            ci_results[metric_name] = f"{metric_value:.3f}±{max(metric_value-ci[0], ci[1]-metric_value):.3f}"

    if cat_mapping:
        reverse_cat_mapping = {v: k for k, v in cat_mapping.items()}
        labels = [reverse_cat_mapping[label] for label in labels]
        results = {reverse_cat_mapping[label]: value for label, value in results.items()}
    
    return ci_results, conf_matrix, results, labels

st.title('결과 비교 애플리케이션')
uploaded_files = st.file_uploader("두 결과 파일을 업로드하세요", accept_multiple_files=True, type=['xlsx'])

if len(uploaded_files) != 2:
    st.error("두 개의 파일을 업로드해야 합니다.")
    st.stop()

if len(uploaded_files) == 2:
    df1 = pd.read_excel(uploaded_files[0])
    df2 = pd.read_excel(uploaded_files[1])
    true_columns = df1.columns.tolist()
    pred_columns = df2.columns.tolist()
    common_columns = [col for col in true_columns if col in pred_columns]

    if not common_columns:
        st.error("두 파일에 공통 컬럼이 없습니다.")
        st.stop()

    data_type_df = pd.DataFrame({'column': common_columns, 'categorical': [infer_column_type(df1[col]) == 'categorical' for col in common_columns]})
    edited_data_type_df = st.data_editor(data_type_df, num_rows="dynamic", key="data_type_editor", width=1200)
    data_types = dict(zip(edited_data_type_df['column'], edited_data_type_df['categorical']))
    common_columns = edited_data_type_df['column'].tolist()
    answer = st.radio("어느 파일이 정답입니까?", (uploaded_files[0].name, uploaded_files[1].name))

    if st.button('결과 비교'):
        results_list = []
        cross_tables = {}

        if answer == uploaded_files[0].name:
            df_true = df1    
            df_pred = df2
            True_data_name = uploaded_files[0].name
            Pred_data_name = uploaded_files[1].name
        else:
            df_true = df2    
            df_pred = df1
            True_data_name = uploaded_files[1].name
            Pred_data_name = uploaded_files[0].name

        for column in common_columns:
            true_data, pred_data, true_type, cat_mapping = validate_and_transform_column(df_true[column], df_pred[column], data_types[column])
            if true_data is not None and pred_data is not None:
                if true_type == 'continuous':
                    if len(true_data) > 0 and len(pred_data) > 0 and len(true_data) == len(pred_data):
                        mse = mean_squared_error(true_data, pred_data)
                        results_list.append({"Label Column": column, "MSE": f"{mse:.2f}"})
                        icc = calculate_icc(true_data, pred_data)
                        results_list.append({"Label Column": column, "ICC": f"{icc:.2f}"})
                    else:
                        st.warning(f"Column '{column}' cannot be processed due to empty or mismatched lengths.")
                elif true_type == 'categorical':
                    if len(true_data) > 0 and len(pred_data) > 0 and len(true_data) == len(pred_data):
                        unique_labels = np.unique(np.concatenate((true_data, pred_data)))
                        metrics, conf_matrix, results, labels = calculate_metrics(true_data, pred_data, unique_labels, cat_mapping)
                        results_list.append({"Label Column": column, **metrics})
                        cross_tables[column] = (conf_matrix, results, labels)
                    else:
                        st.warning(f"Column '{column}' cannot be processed due to empty or mismatched lengths.")
            else:
                st.warning(f"Column '{column}' cannot be processed due to inappropriate data type.")

        if results_list:
            results = pd.DataFrame(results_list)
            st.table(results)
            current_datetime = datetime.now().strftime("%Y%m%d%H%M%S")
            # output_file = Excel_Generator(results, cross_tables)
            output_file = Excel_Generator(results, cross_tables, True_data_name, Pred_data_name)
            output_file_name = f"{os.path.splitext(uploaded_files[0].name)[0]}_{os.path.splitext(uploaded_files[1].name)[0]}_비교분석결과_{current_datetime}.xlsx"
            st.download_button(
                label="Download Output Excel",
                data=output_file,
                file_name=output_file_name,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

            # Cross table 출력
            st.subheader("Cross Tables 및 분수 계산")
            for column, (conf_matrix, results, labels) in cross_tables.items():
                st.write(f"**{column}**")
                st.write("Cross Table:")
                # print(labels)
                st.table(pd.DataFrame(conf_matrix, index=labels, columns=labels))
                st.write("TP, FP, FN, TN:")
                results_df = pd.DataFrame(results)

                # 정밀도 계산 및 추가
                precision_values = results_df.loc['TP'] / (results_df.loc['TP'] + results_df.loc['FP'])
                precision_values = precision_values.fillna(0)  # 나눗셈 결과 NaN 대신 0으로 채우기
                results_df.loc['Precision = TP/(TP+FP)'] = precision_values

                # 재현율 계산 및 추가
                recall_values = results_df.loc['TP'] / (results_df.loc['TP'] + results_df.loc['FN'])
                recall_values = recall_values.fillna(0)  # 나눗셈 결과 NaN 대신 0으로 채우기
                results_df.loc['Recall = TP/(TP+FN)'] = recall_values

                # F1 스코어 계산 및 추가
                f1_scores = 2 * (precision_values * recall_values) / (precision_values + recall_values)
                f1_scores = f1_scores.fillna(0)  # 나눗셈 결과 NaN 대신 0으로 채우기
                results_df.loc['F1 Score'] = f1_scores
               
                st.table(results_df)

                # DataFrame에서 TP, FP, FN, TN 값 추출 및 계산
                TP_list = results_df.loc['TP']
                precision_list = results_df.loc['Precision = TP/(TP+FP)']
                recall_list = results_df.loc['Recall = TP/(TP+FN)']
                f1_list = results_df.loc['F1 Score']

                # 평균 정밀도와 재현율
                TP_sum = np.sum(TP_list)
                Total_n = np.sum(conf_matrix)
                avg_accuracy = TP_sum / Total_n
                avg_precision = np.mean(precision_list)
                avg_recall = np.mean(recall_list)
                avg_f1 = np.mean(f1_list)

                st.write(f"최종 Accuracy (모든 클래스의 TP 합 / 총 데이터 수): {TP_sum} / {Total_n}) = {avg_accuracy:.3f}")
                st.write(f"최종 Precision (Macro방식 - 각 클래스의 Precision의 산술평균): {avg_precision:.3f}")
                st.write(f"최종 Recall (Macro방식 - 각 클래스의 Recall의 산술평균): {avg_recall:.3f}")
                st.write(f"최종 F1 Score (Macro방식 - 각 클래스의 F1 Score의 산술평균): {avg_f1:.3f}")

        else:
            st.write("분석에 적합한 컬럼을 찾을 수 없습니다.")

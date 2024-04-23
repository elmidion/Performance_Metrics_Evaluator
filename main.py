import streamlit as st
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error
import io
import os
from datetime import datetime

def Excel_Generator(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Output')
    output.seek(0)  # 스트림 위치를 처음으로 이동
    processed_data = output.getvalue()
    return processed_data

def infer_column_type(data):
    if pd.api.types.is_numeric_dtype(data):
        return 'continuous'
    elif pd.api.types.is_categorical_dtype(data) or data.dtype == object:
        return 'categorical'
    else:
        return 'unknown'

def validate_and_transform_column(data, is_categorical):
    if is_categorical:
        data = pd.Categorical(data).codes
        return data[data != -1], 'categorical'
    else:
        return data.astype(float).dropna(), 'continuous'

# Streamlit 페이지 설정
st.title('결과 비교 애플리케이션')

# 파일 업로드 섹션
uploaded_files = st.file_uploader("두 결과 파일을 업로드하세요", accept_multiple_files=True, type=['xlsx'])

if len(uploaded_files) != 2:
    st.error("두 개의 파일을 업로드해야 합니다.")
    st.stop()

# 파일이 제대로 업로드 되었다면 
if len(uploaded_files) == 2:
    df1 = pd.read_excel(uploaded_files[0])
    df2 = pd.read_excel(uploaded_files[1])

    # 공통 컬럼 확인
    true_columns = df1.columns.tolist()
    pred_columns = df2.columns.tolist()
    common_columns = [col for col in true_columns if col in pred_columns]

    if not common_columns:
        st.error("두 파일에 공통 컬럼이 없습니다.")
        st.stop()

    # 데이터 형식 확인 및 수정을 위한 DataFrame 생성
    data_type_df = pd.DataFrame({'column': common_columns, 'categorical': [infer_column_type(df1[col]) == 'categorical' for col in common_columns]})

    # 데이터 형식 확인 및 수정
    st.subheader("컬럼 데이터 형식 확인")
    edited_data_type_df = st.data_editor(data_type_df, num_rows="dynamic", key="data_type_editor", width=1200)

    # 수정된 데이터 형식 적용
    data_types = dict(zip(edited_data_type_df['column'], edited_data_type_df['categorical']))
    common_columns = edited_data_type_df['column'].tolist()

    # 사용자에게 어떤 파일이 정답인지 선택하게 함
    answer = st.radio("어느 파일이 정답입니까?", (uploaded_files[0].name, uploaded_files[1].name))

    if st.button('결과 비교'):
        results_list = []

        # 정답 데이터와 비교 데이터 설정
        df_true = df1 if answer == uploaded_files[0].name else df2
        df_pred = df2 if answer == uploaded_files[0].name else df1

        # 공통된 각 라벨 열에 대한 메트릭 계산
        for column in common_columns:
            true_data, true_type = validate_and_transform_column(df_true[column], data_types[column])
            pred_data, pred_type = validate_and_transform_column(df_pred[column], data_types[column])

            if true_data is not None and pred_data is not None:
                if true_type == 'continuous':
                    if len(true_data) > 0 and len(pred_data) > 0 and len(true_data) == len(pred_data):
                        # MSE 계산
                        mse = mean_squared_error(true_data, pred_data)
                        results_list.append({
                            "Label Column": column,
                            "MSE": f"{mse:.2f}"
                        })
                    else:
                        st.warning(f"Column '{column}'은(는) 빈 배열이거나 길이가 일치하지 않아 MSE를 계산할 수 없습니다.")
                elif true_type == 'categorical':
                    if len(true_data) > 0 and len(pred_data) > 0 and len(true_data) == len(pred_data):
                        # Metrics 계산
                        accuracy = accuracy_score(true_data, pred_data)
                        precision = precision_score(true_data, pred_data, average='macro', zero_division=0)
                        recall = recall_score(true_data, pred_data, average='macro', zero_division=0)
                        f1 = f1_score(true_data, pred_data, average='macro', zero_division=0)
                        results_list.append({
                            "Label Column": column,
                            "Accuracy": f"{accuracy:.2f}",
                            "Precision": f"{precision:.2f}",
                            "Recall": f"{recall:.2f}",
                            "F1 Score": f"{f1:.2f}"
                        })
                    else:
                        st.warning(f"Column '{column}'은(는) 빈 배열이거나 길이가 일치하지 않아 분류 메트릭을 계산할 수 없습니다.")
            else:
                st.warning(f"Column '{column}'은(는) 적절한 데이터 유형이 아니므로 처리할 수 없습니다.")

        # 모든 결과를 DataFrame으로 변환
        if results_list:
            results = pd.DataFrame(results_list)
            
            # 'Label Column'을 기준으로 행 순서 정렬
            results['sort_order'] = results['Label Column'].apply(lambda x: common_columns.index(x) if x in common_columns else len(common_columns))
            results = results.sort_values('sort_order').drop('sort_order', axis=1)
            
            st.table(results)

            current_datetime = datetime.now().strftime("%Y%m%d%H%M%S")

            output_file = Excel_Generator(results)
            output_file_name = f"{os.path.splitext(uploaded_files[1].name)[0]}_비교분석결과_{current_datetime}.xlsx"

            st.download_button(
                label="Download Output Excel",
                data=output_file,
                file_name=output_file_name,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        else:
            st.write("분석에 적합한 컬럼을 찾을 수 없습니다.")
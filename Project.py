import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from transformers import pipeline
from deep_translator import GoogleTranslator
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from vnstock import Vnstock
import concurrent.futures
import re

API_URL = "https://cafef.vn/du-lieu/ajax/mobile/smart/ajaxhanghoa.ashx?type="

def fetch_data(data_type):
    url = f"{API_URL}{data_type}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json().get("Data", [])
    return []

@st.cache_data
def load_esg_data():
    url = "https://docs.google.com/spreadsheets/d/1OeA2H9VNltu7hnhSR_U-lFBNUE_lJ_qq/export?format=xlsx"
    df = pd.read_excel(url)
    df['Code'] = df['Code'].str.split('.').str[0]  # Chuẩn hóa mã cổ phiếu
    return df

df_esg = load_esg_data()

def draw_circle(color, text):
    fig, ax = plt.subplots(figsize=(4, 4), dpi=200)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    ax.set_axis_off()

    circle = plt.Circle((0, 0), 1, color=color)
    ax.add_patch(circle)
    ax.text(0, 0, text, fontsize=20, color="white", ha="center", va="center", fontweight="bold")

    return fig

@st.cache_resource
def load_finbert():
    return pipeline("text-classification", model="ProsusAI/finbert")

sentiment_pipeline = load_finbert()

def translate_to_english(text):
    try:
        return GoogleTranslator(source='auto', target='en').translate(text)
    except:
        return text  

def sentiment_to_score(label, score):
    if label == "positive":
        return score * 100
    elif label == "negative":
        return (1 - score) * 100
    else:
        return 50

def classify_sentiment(score):
    if score < 33:
        return "❌ Tiêu cực"
    elif 33 <= score < 66:
        return "⚠ Trung lập"
    else:
        return "✅ Tích cực"

start_date = datetime(2015, 1, 1).strftime('%Y-%m-%d')
end_date = datetime.today().strftime('%Y-%m-%d')

stock = Vnstock().stock(symbol='ACB', source='TCBS')

@st.cache_data
def get_vnindex_data():
    df = stock.quote.history(symbol='VNINDEX', start=start_date, end=end_date, interval='1D')
    if df is not None and not df.empty:
        df = df.dropna().sort_values(by='time')
        df['time'] = pd.to_datetime(df['time'])
    return df

df_vnindex = get_vnindex_data()

def create_vnindex_chart(df):
    fig = go.Figure()

    df['time'] = pd.to_datetime(df['time']).dt.strftime('%Y-%m-%d')

    fig.add_trace(go.Scatter(
        x=df['time'], y=df['close'], mode='lines', name='Giá đóng cửa',
        line=dict(color='blue')
    ))

    fig.add_trace(go.Bar(
        x=df['time'], y=df['volume'], name='Khối lượng giao dịch',
        marker_color='green', opacity=0.5, yaxis='y2'
    ))

    tick_indices = list(range(0, len(df), max(len(df) // 10, 1)))
    tick_labels = [df['time'].iloc[i] for i in tick_indices]

    fig.update_layout(
        title='📈 Giá đóng cửa & Khối lượng Giao dịch VNINDEX (Nguồn: Vnstock)',
        xaxis_title='Thời gian',
        yaxis_title='Giá đóng cửa',
        yaxis2=dict(
            title='Khối lượng giao dịch',
            overlaying='y',
            side='right',
            showgrid=False
        ),
        xaxis=dict(
            type="category",
            tickmode='array',
            tickvals=tick_labels,
            ticktext=tick_labels,
            tickangle=-45
        ),
        bargap=0,
        xaxis_rangeslider_visible=False,
        showlegend=True
    )
    return fig

def get_stock_data(symbol):
    stock = Vnstock().stock(symbol=symbol, source='TCBSTCBS')  
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=365*5)).strftime('%Y-%m-%d')  
    df = stock.quote.history(symbol=symbol, start=start_date, end=end_date, interval='1D')
    df = df.dropna().sort_values(by='time')
    df = df[df['volume'] > 0]  
    df.reset_index(drop=True, inplace=True)
    return df

def create_line_chart(df, symbol):
    fig = go.Figure()
    df['time'] = pd.to_datetime(df['time']).dt.strftime('%Y-%m-%d')
    fig.add_trace(go.Scatter(
        x=df['time'], y=df['close'], mode='lines', name='Giá thực tế',
        line=dict(color='blue')
    ))

    fig.add_trace(go.Bar(
        x=df['time'], y=df['volume'], name='Khối lượng giao dịch',
        marker_color='green', opacity=0.5, yaxis='y2'
    ))

    tick_indices = list(range(0, len(df), max(len(df) // 10, 1)))
    tick_labels = [df['time'].iloc[i] for i in tick_indices]

    fig.update_layout(
        title=f'Đồ thị giá đóng cửa và khối lượng của mã: {symbol} (Nguồn: Vnstock)',
        xaxis_title='Date',
        yaxis_title='Price (VNĐ)',
        yaxis2=dict(
            title='Volume',
            overlaying='y',
            side='right',
            showgrid=False
        ),
        xaxis=dict(
            type="category",
            tickmode='array',
            tickvals=tick_labels,
            ticktext=tick_labels,
            tickangle=-45
        ),
        bargap=0,
        xaxis_rangeslider_visible=False,
        showlegend=True
    )
    return fig

def get_latest_articles(symbol, limit=20):
    data_rows = []
    try:
        url = f'https://s.cafef.vn/Ajax/Events_RelatedNews_New.aspx?symbol={symbol}&PageIndex=1&PageSize={limit}&Type=2'
        headers = {'User-Agent': 'Mozilla/5.0'}
        r = requests.get(url, headers=headers)
        soup = BeautifulSoup(r.content, "html.parser")
        data = soup.find("ul", {"class": "News_Title_Link"})

        if not data:
            return pd.DataFrame()

        for row in data.find_all('li'):
            news_date = row.span.text.strip()
            title = row.a.text.strip()
            article_url = "https://s.cafef.vn/" + str(row.a['href'])
            data_rows.append({"news_date": news_date, "title": title, "url": article_url, "symbol": symbol})
            if len(data_rows) >= limit:
                break
    except:
        return pd.DataFrame()

    return pd.DataFrame(data_rows)

BASE_URL = "https://vnexpress.net"
SEARCH_URL = "https://vnexpress.net/kinh-doanh/vi-mo"

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

def extract_publish_date(url):
    """Trích xuất ngày đăng từ một bài báo cụ thể trên VNExpress."""
    try:
        response = requests.get(url, headers=headers, timeout=5)
        if response.status_code != 200:
            return "Không xác định"

        soup = BeautifulSoup(response.text, 'html.parser')
        date_element = soup.find("span", class_="date")
        if date_element:
            date_match = re.search(r'(\d{1,2}/\d{1,2}/\d{4})', date_element.text)
            if date_match:
                return date_match.group(1)
    except Exception as e:
        print(f"Lỗi khi lấy ngày từ {url}: {e}")
        return "Không xác định"

    return "Không xác định"

def get_macro_news():
    """Lấy danh sách tin tức vĩ mô từ VNExpress và cập nhật ngày đăng chính xác."""
    news_data = []
    page = 1

    while len(news_data) < 20:
        response = requests.get(f"{SEARCH_URL}-p{page}", headers=headers, timeout=20)
        if response.status_code != 200:
            break

        soup = BeautifulSoup(response.text, 'html.parser')
        articles = soup.find_all("p", class_="description")

        links = []
        for article in articles:
            if len(news_data) >= 20:
                break
            try:
                a_tag = article.find("a", attrs={"data-medium": True, "href": True, "title": True})
                if not a_tag:
                    continue
                title = a_tag.get("title").strip()
                link = a_tag.get("href").strip()
                if not link.startswith("http"):
                    link = BASE_URL + link
                description = article.text.strip()
                links.append((title, description, link))
            except:
                continue

        with concurrent.futures.ThreadPoolExecutor() as executor:
            dates = executor.map(extract_publish_date, [link for _, _, link in links])

        for (title, description, link), article_date in zip(links, dates):
            news_data.append({"title": title, "date": article_date, "description": description, "url": link})

        page += 1

    return pd.DataFrame(news_data)

def analyze_sentiment(df):
    if df.empty:
        return df

    df["Analysis_Text"] = df["title"] + ". " + df.get("description", "")
    df["Translated_Text"] = df["Analysis_Text"].apply(translate_to_english)
    sentiments_translated = sentiment_pipeline(df["Translated_Text"].tolist())
    df["Sentiment_Score"] = [sentiment_to_score(res["label"], res["score"]) for res in sentiments_translated]
    df["Sentiment_Label"] = df["Sentiment_Score"].apply(classify_sentiment)
    return df

st.title("📊 Phân tích Cổ phiếu & Tin tức Vĩ mô")

selected_mode = st.radio(
    "Chọn loại dữ liệu:",
    ["Dữ liệu Doanh nghiệp", "Dữ liệu Vĩ mô", "Tổng quan thị trường"],
    key="unique_data_mode_selector"
)

if selected_mode == "Dữ liệu Doanh nghiệp":
    analysis_type = st.selectbox(
        "Chọn loại phân tích:",
        ["Phân tích giá cổ phiếu & Tin tức", "Phân tích Tài chính Cổ phiếu", "Phân tích ESG"],
        key="enterprise_analysis_type"
    )

    stock_code = st.text_input("Nhập mã cổ phiếu (VD: ACB, HPG, VNM):").upper()

    if analysis_type == "Phân tích giá cổ phiếu & Tin tức" and stock_code:
        df_stock = get_stock_data(stock_code)
        if not df_stock.empty:
            st.plotly_chart(create_line_chart(df_stock, stock_code))

        df_news = get_latest_articles(stock_code, limit=20)
        if not df_news.empty:
            df_news = analyze_sentiment(df_news)
            st.write("### 📰 Tin Tức Doanh Nghiệp & Phân Tích Cảm Xúc")
            st.dataframe(df_news[['news_date', 'title', 'Sentiment_Score', 'Sentiment_Label', 'url']])

            avg_sentiment = df_news["Sentiment_Score"].mean()
            st.plotly_chart(go.Figure(go.Indicator(
                mode="gauge+number",
                value=avg_sentiment,
                title={'text': f"Cảm xúc tổng thể tin tức {stock_code}", 'font': {'size': 24}},
                gauge={'axis': {'range': [0, 100]},
                       'steps': [{'range': [0, 33], 'color': '#FF4C4C'},
                                 {'range': [33, 66], 'color': '#FFDD57'},
                                 {'range': [66, 100], 'color': '#4CAF50'}]})))

    elif analysis_type == "Phân tích ESG" and stock_code:
        result = df_esg[df_esg['Code'] == stock_code]

        if result.empty:
            st.pyplot(draw_circle("red", "Không công bố ESG"))
        else:
            esg_score = result["ESG Combined Score"].mean()
            st.pyplot(draw_circle("green", f"{esg_score:.2f}"))

    elif analysis_type == "Phân tích Tài chính Cổ phiếu" and stock_code:
        stock = Vnstock().stock(symbol=stock_code, source='VCI')

        df_balancesheet = stock.finance.balance_sheet(period='year', lang='vi', dropna=True)
        if df_balancesheet is None or df_balancesheet.empty:
            st.error("🚨 Không thể tải dữ liệu Bảng Cân Đối Kế Toán!")
            st.stop()

        balance_columns = ['Năm', 'TỔNG CỘNG TÀI SẢN (Tỷ đồng)', 'NỢ PHẢI TRẢ (Tỷ đồng)']
        df_balancesheet = df_balancesheet[balance_columns]
        df_balancesheet['Năm'] = df_balancesheet['Năm'].astype(int)
        df_balancesheet = df_balancesheet.sort_values(by='Năm', ascending=False).head(5)

        st.subheader(f"📉 Tài Sản & Nợ Phải Trả (Nguồn: Vnstock)")
        fig1 = px.bar(df_balancesheet, x='Năm', y=balance_columns[1:],
                    barmode='group', title="Tài Sản & Nợ Phải Trả",
                    labels={'value': "Tỷ đồng", 'variable': "Chỉ tiêu"})
        fig1.for_each_trace(lambda t: t.update(name=t.name.replace(" (Tỷ đồng)", "")))
        fig1.update_layout(yaxis_title="Nghìn tỷ đồng")
        st.plotly_chart(fig1)

        df_income = stock.finance.income_statement(period='year', lang='vi', dropna=True)
        if df_income is None or df_income.empty:
            st.error("🚨 Không thể tải dữ liệu Báo Cáo Thu Nhập!")
            st.stop()

        income_columns = ['Năm', 'Doanh thu (Tỷ đồng)', 'Lợi nhuận thuần']
        df_income = df_income[income_columns]
        df_income['Năm'] = df_income['Năm'].astype(int)
        df_income = df_income.sort_values(by='Năm', ascending=False).head(5)

        st.subheader(f"📈 Doanh Thu & Lợi Nhuận (Nguồn: Vnstock)")
        fig2 = px.bar(df_income, x='Năm', y=income_columns[1:],
                    barmode='group', title="Doanh Thu & Lợi Nhuận",
                    labels={'value': "Tỷ đồng", 'variable': "Chỉ tiêu"})
        fig2.for_each_trace(lambda t: t.update(name=t.name.replace(" (Tỷ đồng)", "")))
        fig2.update_layout(yaxis_title="Nghìn tỷ đồng")
        st.plotly_chart(fig2)

        df_cash = stock.finance.cash_flow(period='year', lang='vi', dropna=True)
        if df_cash is None or df_cash.empty:
            st.error("🚨 Không thể tải dữ liệu Dòng Tiền!")
            st.stop()

        cash_columns = ['Năm', 'Lưu chuyển từ hoạt động đầu tư',
                        'Lưu chuyển tiền từ hoạt động tài chính',
                        'Lưu chuyển tiền tệ ròng từ các hoạt động SXKD']
        df_cash = df_cash[cash_columns]
        df_cash['Năm'] = df_cash['Năm'].astype(int)
        df_cash = df_cash.sort_values(by='Năm', ascending=False).head(5)

        st.subheader(f"💰 Dòng Tiền (Nguồn: Vnstock)")
        fig3 = px.bar(df_cash, x='Năm', y=cash_columns[1:],
                    barmode='group', title="Dòng Tiền",
                    labels={'value': "Tỷ đồng", 'variable': "Chỉ tiêu"})
        fig3.for_each_trace(lambda t: t.update(name=t.name.replace(" (Tỷ đồng)", "")))
        fig3.update_layout(yaxis_title="Nghìn tỷ đồng")
        st.plotly_chart(fig3)

        df_ratio = stock.finance.ratio(period='year', lang='vi', dropna=True)
        if df_ratio is None or df_ratio.empty:
            st.error("🚨 Không thể tải dữ liệu Chỉ Số Tài Chính!")
            st.stop()

        ratio_columns = [
            ('Meta', 'Năm'),
            ('Chỉ tiêu khả năng sinh lợi', 'ROE (%)'),
            ('Chỉ tiêu khả năng sinh lợi', 'ROA (%)'),
            ('Chỉ tiêu định giá', 'EPS (VND)')
        ]
        df_ratio = df_ratio[ratio_columns]
        df_ratio[('Meta', 'Năm')] = df_ratio[('Meta', 'Năm')].astype(int)

        df_ratio[('Chỉ tiêu khả năng sinh lợi', 'ROE (%)')] *= 100
        df_ratio[('Chỉ tiêu khả năng sinh lợi', 'ROA (%)')] *= 100
        
        df_ratio = df_ratio.sort_values(by=('Meta', 'Năm'), ascending=False).head(5)
        df_ratio.columns = ['Năm', 'ROE (%)', 'ROA (%)', 'EPS (VND)']

        st.subheader("📊 ROE & ROA (Nguồn: Vnstock)")
        fig4 = px.bar(df_ratio, x='Năm', y=['ROE (%)', 'ROA (%)'],
                    barmode='group', title="ROE & ROA",
                    labels={'value': "Tỷ lệ", 'variable': "Chỉ tiêu"})
        fig4.for_each_trace(lambda t: t.update(name=t.name.replace(" (Tỷ đồng)", "")))
        fig4.update_layout(yaxis_title="Tỷ lệ (%)")
        st.plotly_chart(fig4)

elif selected_mode == "Dữ liệu Vĩ mô":
    st.plotly_chart(create_vnindex_chart(df_vnindex))

    df_macro = get_macro_news()

    if not df_macro.empty:  
        df_macro = analyze_sentiment(df_macro)

        st.write("### 🌍 Tin Tức Vĩ Mô & Phân tích cảm xúc bằng mô hình Finbert (Nguồn: VnExpress)")
        for index, row in df_macro.iterrows():
            st.markdown(f"**🗓 Date**: {row['date']} | **📰 Title**: [{row['title']}]({row['url']})")
            st.markdown(f"📊 **Sentiment Score**: {row['Sentiment_Score']:.2f} - {row['Sentiment_Label']}")

        average_sentiment = df_macro["Sentiment_Score"].mean()
        st.plotly_chart(go.Figure(go.Indicator(
            mode="gauge+number",
            value=average_sentiment,
            title={'text': "Điểm cảm xúc trung bình", 'font': {'size': 24}},
            gauge={'axis': {'range': [0, 100]},
                   'steps': [{'range': [0, 33], 'color': '#FF4C4C'},
                             {'range': [33, 66], 'color': '#FFDD57'},
                             {'range': [66, 100], 'color': '#4CAF50'}]})))

        sentiment_counts = df_macro['Sentiment_Score'].apply(
            lambda x: "POSITIVE" if x > 66 else "NEUTRAL" if x > 33 else "NEGATIVE"
        ).value_counts()
        sentiment_df = pd.DataFrame({'Sentiment': sentiment_counts.index, 'Count': sentiment_counts.values})
        fig_plotly = px.bar(sentiment_df, x='Sentiment', y='Count', color='Sentiment',
                            color_discrete_map={'NEGATIVE': 'red', 'NEUTRAL': 'yellow', 'POSITIVE': 'green'},
                            title="Sentiment Distribution")
        st.plotly_chart(fig_plotly)

        text_data = " ".join(df_macro['title'])
        wordcloud = WordCloud(width=800, height=400, background_color='black', colormap='viridis').generate(text_data)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)

    st.title("📊 Dữ liệu thị trường tài chính hôm nay (Nguồn: CafeF)")

    data_type = st.selectbox("🔎 Chọn danh mục", ["Hàng hóa", "Tỷ giá", "Tiền mã hóa"])

    type_mapping = {"Hàng hóa": 1, "Tỷ giá": 2, "Tiền mã hóa": 3}
    selected_type = type_mapping[data_type]

    data = fetch_data(selected_type)

    if data:
        if selected_type == 1:
            df = pd.DataFrame(data)[["goods", "last", "changePercent"]]
            df.rename(columns={"goods": "Tên", "last": "Giá", "changePercent": "Thay đổi (%)"}, inplace=True)


        elif selected_type == 2:
            df = pd.DataFrame(data)[["ProductName", "CurrentPrice", "change24H"]]
            df.rename(columns={"ProductName": "Tên", "CurrentPrice": "Giá", "change24H": "Thay đổi (%)"}, inplace=True)
        elif selected_type == 3:
            df = pd.DataFrame(data)[["name", "price", "change24H"]]
            df.rename(columns={"name": "Tên", "price": "Giá", "change24H": "Thay đổi (%)"}, inplace=True)

        st.dataframe(df)
    else:
        st.warning("⚠️ Không có dữ liệu!")

elif selected_mode == "Tổng quan thị trường":
    today = datetime.today().strftime('%d-%m-%Y')
    st.title(f"📊 TOÀN CẢNH THỊ TRƯỜNG - NGÀY: {today}")

    STOCK_API_URL = "https://cafef.vn/du-lieu/Ajax/Mobile/Smart/AjaxTop10CP.ashx?centerID={}&type={}"

    MARKET_MAP = {
        "HOSE": "HOSE",
        "HNX": "HNX",
        "VN30": "VN30"
    }
    TYPE_MAP = {
        "Tăng giá (UP)": "UP",
        "Giảm giá (DOWN)": "DOWN",
        "Khối lượng giao dịch (VOLUME)": "VOLUME"
    }

    def fetch_stock_data(market, data_type):
        url = STOCK_API_URL.format(market, data_type)
        response = requests.get(url)
        if response.status_code == 200:
            try:
                data = response.json().get("Data", [])
                processed_data = [
                    {
                        "STT": idx + 1,
                        "Mã CK": item["Symbol"],
                        "KL mua ròng": item["Volume"],
                        "Giá": item["CurrentPrice"],
                        "Thay đổi": f"{item['ChangePrice']} ({item['ChangePricePercent']}%)"
                    }
                    for idx, item in enumerate(data)
                ]
                return pd.DataFrame(processed_data)
            except Exception as e:
                st.error(f"Lỗi xử lý dữ liệu: {e}")
                return None
        else:
            st.error("Không thể lấy dữ liệu từ API")
            return None

    st.title("Top 10 cổ phiếu (Nguồn: CafeF)")
    selected_market = st.selectbox("Chọn sàn giao dịch", list(MARKET_MAP.keys()), index=0)
    selected_type = st.selectbox("Chọn loại dữ liệu", list(TYPE_MAP.keys()), index=0)

    market_code = MARKET_MAP[selected_market]
    type_code = TYPE_MAP[selected_type]
    df = fetch_stock_data(market_code, type_code)
    if df is not None and not df.empty:
        st.dataframe(df)
    else:
        st.warning("Không có dữ liệu để hiển thị.")

    API_URL = "https://api-finance-t19.24hmoney.vn/v2/ios/company-group/all-level-with-summary?device_id=web17406339ikhn53nn5up2uqzfk91s8066yte1q5a456253&device_name=INVALID&device_model=Windows+11&network_carrier=INVALID&connection_type=INVALID&os=Chrome&os_version=133.0.0.0&access_token=INVALID&push_token=INVALID&locale=vi&browser_id=web17406339ikhn53nn5up2uqzfk91s8066yte1q5a456253&type=day"

    headers = {"User-Agent": "Mozilla/5.0"}

    def get_stock_data():
        response = requests.get(API_URL, headers=headers)
        if response.status_code == 200:
            data = response.json()
            records = []

            def extract_data(json_data, parent_name=""):
                for group in json_data:
                    total_val = group.get("total_val", 0.0)
                    total_val_increase = group.get("total_val_increase", 0)
                    total_val_nochange = group.get("total_val_nochange", 0)
                    total_val_decrease = group.get("total_val_decrease", 0)

                    if total_val == 0:
                        pct_increase, pct_nochange, pct_decrease = 0, 0, 0
                    else:
                        pct_increase = (total_val_increase / total_val) * 100
                        pct_nochange = (total_val_nochange / total_val) * 100
                        pct_decrease = (total_val_decrease / total_val) * 100

                    records.append({
                        "icb_code": group.get("icb_code", ""),
                        "icb_name": group.get("icb_name", ""),
                        "icb_level": group.get("icb_level", ""),
                        "parent_name": parent_name,
                        "total_stock": group.get("total_stock", 0),
                        "total_stock_increase": group.get("total_stock_increase", 0),
                        "total_stock_nochange": group.get("total_stock_nochange", 0),
                        "total_stock_decrease": group.get("total_stock_decrease", 0),
                        "avg_change_percent": group.get("avg_change_percent", 0),
                        "total_val": total_val,
                        "total_val_increase": total_val_increase,
                        "total_val_nochange": total_val_nochange,
                        "total_val_decrease": total_val_decrease,
                        "pct_increase": pct_increase,  
                        "pct_nochange": pct_nochange,  
                        "pct_decrease": pct_decrease    
                    })

                    if "child" in group and isinstance(group["child"], list):
                        extract_data(group["child"], parent_name=group["icb_name"])

            extract_data(data["data"]["groups"])

            df = pd.DataFrame(records)

            return df
        else:
            st.error(f"❌ Lỗi khi lấy dữ liệu từ API: {response.status_code}")
            return pd.DataFrame()

    df = get_stock_data()
    df_filtered = df[df["icb_level"] == 2]

    st.title("📊 Thống kê ngành (Nguồn: 24h Money)")

    if not df_filtered.empty:
        st.subheader("📌 Danh sách ngành")

        def format_percent(val):
            color = "red" if val < 0 else "green"
            return f'<span style="color:{color}; font-weight:bold">{val:.2f}%</span>'

        df_display = df_filtered[[
            "icb_name", "avg_change_percent", "total_val",
            "total_stock_increase", "total_stock_nochange", "total_stock_decrease"
        ]].copy()

        df_display.columns = [
            "Ngành", "Biến động giá (%)", "Giá trị GD (tỷ)",
            "Số lượng cổ phiếu tăng", "Số lượng cổ phiếu không đổi", "Số lượng cổ phiếu giảm"
        ]

        df_display["Biến động giá (%)"] = df_display["Biến động giá (%)"].apply(format_percent)
        df_display["Giá trị GD (tỷ)"] = df_display["Giá trị GD (tỷ)"].astype(float)

        df_display = df_display.sort_values(by="Giá trị GD (tỷ)", ascending=False)

        st.markdown(df_display.to_html(escape=False, index=False), unsafe_allow_html=True)

        fig = go.Figure()

        fig.add_trace(go.Bar(
            y=df_filtered["icb_name"],
            x=df_filtered["pct_increase"],
            name="Tăng",
            orientation="h",
            marker=dict(color="green"),
            hovertemplate="<b>%{y}</b><br>🔼 Giá trị tăng: %{customdata} tỷ<extra></extra>",
            customdata=df_filtered["total_val_increase"]  
        ))

        fig.add_trace(go.Bar(
            y=df_filtered["icb_name"],
            x=df_filtered["pct_nochange"],
            name="Không đổi",
            orientation="h",
            marker=dict(color="yellow"),
            hovertemplate="<b>%{y}</b><br>⚖️ Giá trị không đổi: %{customdata} tỷ<extra></extra>",
            customdata=df_filtered["total_val_nochange"]  
        ))

        fig.add_trace(go.Bar(
            y=df_filtered["icb_name"],
            x=df_filtered["pct_decrease"],
            name="Giảm",
            orientation="h",
            marker=dict(color="red"),
            hovertemplate="<b>%{y}</b><br>🔻 Giá trị giảm: %{customdata} tỷ<extra></extra>",
            customdata=df_filtered["total_val_decrease"]  
        ))

        fig.update_layout(
            title="📌 Phân bổ dòng tiền theo ngành (Nguồn: 24h Money)",
            xaxis_title="Phân bổ dòng tiền",
            yaxis_title="Ngành",
            barmode="stack", 
            xaxis=dict(showticklabels=False),  
            xaxis_tickangle=-45
        )

        st.plotly_chart(fig)

    else:
        st.warning("Không có dữ liệu để hiển thị.")

    category_mapping = {
        "Tất cả các ngành": 0,
        "Bất động sản và Xây dựng": 345,
        "Công nghệ": 347,
        "Công nghiệp": 343,
        "Dịch vụ": 346,
        "Hàng tiêu dùng": 339,
        "Năng lượng": 340,
        "Nguyên vật liệu": 344,
        "Nông nghiệp": 338,
        "Tài chính": 341,
        "Viễn thông": 348,
        "Y tế": 342
    }

    center_mapping = {
        "HOSE": 1,
        "HNX": 2,
        "UPCoM": 9
    }

    def get_cafef_data(category_id, center_id):
        url = f"https://cafef.vn/du-lieu/ajax/mobile/smart/ajaxbandothitruong.ashx?type=1&category={category_id}&centerId={center_id}"
        headers = {"User-Agent": "Mozilla/5.0"}

        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            try:
                data = response.json()
                return pd.DataFrame(data["Data"])
            except Exception as e:
                st.error(f"❌ Lỗi khi xử lý JSON: {e}")
        else:
            st.error(f"❌ Lỗi khi lấy dữ liệu từ CafeF: {response.status_code}")
        return pd.DataFrame()

    def get_foreign_trading_data():
        url = "https://api-finance-t19.24hmoney.vn/v2/web/indices/foreign-trading-all-stock-by-time?code=10&type=today"
        headers = {"User-Agent": "Mozilla/5.0"}

        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            try:
                data = response.json()
                return pd.DataFrame(data.get("data", {}).get("data", []))
            except Exception as e:
                st.error(f"❌ Lỗi khi xử lý JSON: {e}")
        else:
            st.error(f"❌ Lỗi khi lấy dữ liệu từ 24hMoney: {response.status_code}")
        return pd.DataFrame()

    st.title("📊 Bản đồ thị trường (Nguồn: CafeF)")

    selected_sector = st.selectbox("🔍 Chọn nhóm ngành", list(category_mapping.keys()))
    selected_center = st.selectbox("📍 Chọn sàn giao dịch", list(center_mapping.keys()))

    category_id = category_mapping[selected_sector]
    center_id = center_mapping[selected_center]
    df_cafef = get_cafef_data(category_id, center_id)

    if not df_cafef.empty:
        st.subheader(f"📌 Bản đồ thị trường - {selected_sector} ({selected_center})")

        color_map = {2: "#800080", 1: "#008000", 0: "#FF0000", -1: "#FFD700"}
        df_cafef["ColorMapped"] = df_cafef["Color"].map(color_map)

        df_cafef["Label"] = df_cafef["Symbol"] + "<br>" + df_cafef["ChangePercent"].astype(str) + "%"

        option = st.selectbox(
            "📊 Chọn tiêu chí vẽ Treemap:",
            ["Khối lượng giao dịch", "Giá trị giao dịch", "Vốn hóa"]
        )

        column_mapping = {"Khối lượng giao dịch": "TotalVolume", "Giá trị giao dịch": "TotalValue", "Vốn hóa": "MarketCap"}
        selected_column = column_mapping[option]

        fig1 = px.treemap(
            df_cafef,
            path=["Label"],
            values=selected_column,
            color="ColorMapped",
            hover_data=["ChangePercent"],
            color_discrete_map=color_map
        )

        st.plotly_chart(fig1)
    else:
        st.warning("Không thể tải dữ liệu từ CafeF.")

    df_foreign = get_foreign_trading_data()
    if not df_foreign.empty:
       
        top_buy = df_foreign.nlargest(10, 'net_val').sort_values('net_val', ascending=True)
        top_sell = df_foreign.nsmallest(10, 'net_val').sort_values('net_val', ascending=False)

        fig2 = go.Figure()

        fig2.add_trace(go.Bar(
            y=top_sell['symbol'],
            x=top_sell['net_val'],
            orientation='h',
            name='Top bán ròng (Tỷ đồng)',
            marker_color='red',
            yaxis='y1'
        ))

        fig2.add_trace(go.Bar(
            y=top_buy['symbol'],
            x=top_buy['net_val'],
            orientation='h',
            name='Top mua ròng (Tỷ đồng)',
            marker_color='green',
            yaxis='y2'
        ))

        fig2.update_layout(
            title="Giao dịch khối ngoại - Top mua ròng và bán ròng (Nguồn: 24h Money)",
            xaxis_title="Giá trị mua/bán ròng (Tỷ đồng)",
            yaxis_title="Mã chứng khoán",
            legend_title="Loại giao dịch",
            barmode='relative',
            xaxis=dict(zeroline=True),
            yaxis=dict(title="Top bán ròng", side="left", showgrid=False),
            yaxis2=dict(title="Top mua ròng", overlaying="y", side="right", showgrid=False)
        )

        st.plotly_chart(fig2)
    else:
        st.warning("Không thể tải dữ liệu từ 24hMoney.")

    MARKET_LEADER_API = "https://msh-appdata.cafef.vn/rest-api/api/v1/MarketLeaderGroup?centerId={}"
    MARKET_MAP = {"VN-Index": 1, "HNX": 2, "UPCOM": 9}

    headers = {"User-Agent": "Mozilla/5.0"}

    def fetch_market_leader_data(market_id):
        """Lấy dữ liệu từ API của CafeF"""
        url = MARKET_LEADER_API.format(market_id)
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            try:
                data = response.json().get("data", [])
                return pd.DataFrame(data)
            except Exception as e:
                st.error(f"Lỗi xử lý dữ liệu: {e}")
        else:
            st.error("Không thể lấy dữ liệu từ API")
        return pd.DataFrame()

    MARKET_LEADER_API = "https://msh-appdata.cafef.vn/rest-api/api/v1/MarketLeaderGroup?centerId={}"
    MARKET_MAP = {"VN-Index": 1, "HNX": 2, "UPCOM": 9}

    headers = {"User-Agent": "Mozilla/5.0"}

    def fetch_market_leader_data(market_id):
        """Lấy dữ liệu từ API của CafeF"""
        url = MARKET_LEADER_API.format(market_id)
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            try:
                data = response.json().get("data", [])
                return pd.DataFrame(data)
            except Exception as e:
                st.error(f"Lỗi xử lý dữ liệu: {e}")
        else:
            st.error("Không thể lấy dữ liệu từ API")
        return pd.DataFrame()

    st.title("📈 Nhóm dẫn dắt thị trường")
    selected_market = st.selectbox("Chọn sàn giao dịch", list(MARKET_MAP.keys()), index=0)
    market_id = MARKET_MAP[selected_market]

    df_market = fetch_market_leader_data(market_id)
    if not df_market.empty:
        
        df_market["color"] = df_market["score"].apply(lambda x: "green" if x > 0 else "red")

        df_market = pd.concat([
            df_market[df_market["score"] > 0].sort_values(by="score", ascending=False),
            df_market[df_market["score"] <= 0].sort_values(by="score", ascending=False)
        ])

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=df_market["symbol"],
            y=df_market["score"],
            marker_color=df_market["color"],
            text=df_market["score"],
            textposition="outside"
        ))

        fig.update_layout(
            title=f"Mức đóng góp của cổ phiếu đến {selected_market} (Nguồn: CafeF)",
            xaxis_title="Mã CK",
            yaxis_title="Mức đóng góp",
            showlegend=False
        )
        st.plotly_chart(fig)
    else:
        st.warning("Không có dữ liệu để hiển thị.")

    today_str = datetime.today().strftime('%Y%m%d')

    API_BASE_URL = "https://msh-appdata.cafef.vn/rest-api/api/v1/OverviewOrgnizaztion/0/{date}/{type}?symbol={symbol}"

    def fetch_data(symbol, transaction_type, date):
        """ Lấy dữ liệu từ API dựa trên mã cổ phiếu, loại giao dịch và ngày """
        url = API_BASE_URL.format(date=date, type=transaction_type, symbol=symbol)
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            df = pd.DataFrame(data)
            if not df.empty:
                df["date"] = df["date"].str[:10]  
                df = df.sort_values(by="date")
            return df
        else:
            st.error(f"❌ Lỗi lấy dữ liệu từ API: {response.status_code}")
            return pd.DataFrame()

    st.title("📊 Phân tích Giao dịch Chứng khoán (Nguồn: CafeF)")

    symbol = st.text_input("Nhập mã cổ phiếu (VD: MWG, HPG, VNM):", value="VNINDEX").upper()

    transaction_options = {
        "Tự doanh": 20,
        "Khối ngoại": 15
    }
    selected_transaction = st.selectbox("🔍 Chọn loại giao dịch", list(transaction_options.keys()))
    transaction_type = transaction_options[selected_transaction]

    data_options = {
        "Khối lượng giao dịch": "volume",
        "Giá trị giao dịch": "value"
    }
    selected_data = st.radio("📊 Chọn loại dữ liệu", list(data_options.keys()))
    data_type = data_options[selected_data]

    df = fetch_data(symbol, transaction_type, today_str)

    if not df.empty:
        df = df[["date", "buyVol", "buyVal", "sellVol", "sellVal", "netVol", "netVal"]]

        df["sellVol"] = -df["sellVol"]
        df["sellVal"] = -df["sellVal"]

        tick_indices = list(range(0, len(df), max(len(df) // 8, 1)))
        tick_labels = [df["date"].iloc[i] for i in tick_indices]

        fig = go.Figure()

        if data_type == "volume":
            
            fig.add_trace(go.Bar(
                x=df["date"], y=df["buyVol"], name="Khối lượng Mua", marker_color="blue"
            ))
            fig.add_trace(go.Bar(
                x=df["date"], y=df["sellVol"], name="Khối lượng Bán", marker_color="red"
            ))
            fig.update_layout(
                title=f"Chi tiết giao dịch mua và bán (Khối lượng) - {symbol}", 
                xaxis_title="Ngày", yaxis_title="Khối lượng",
                xaxis=dict(
                    type="category",
                    tickmode="array",
                    tickvals=tick_labels,
                    ticktext=tick_labels
                ),
                barmode="relative"
            )

        elif data_type == "value":
            
            fig.add_trace(go.Bar(
                x=df["date"], y=df["buyVal"], name="Giá trị Mua", marker_color="blue"
            ))
            fig.add_trace(go.Bar(
                x=df["date"], y=df["sellVal"], name="Giá trị Bán", marker_color="red"
            ))
            fig.update_layout(
                title=f"Chi tiết giao dịch mua và bán (Giá trị) - {symbol}", 
                xaxis_title="Ngày", yaxis_title="Giá trị (Tỷ đồng)",
                xaxis=dict(
                    type="category",
                    tickmode="array",
                    tickvals=tick_labels,
                    ticktext=tick_labels
                ),
                barmode="relative"
            )

        st.plotly_chart(fig)
    else:
        st.warning("⚠ Không có dữ liệu!")

API_BASE_URL = "https://msh-appdata.cafef.vn/rest-api/api/v1/Liquidity/{symbol}"

def fetch_liquidity_data(symbol):
    """Lấy dữ liệu thanh khoản từ API dựa trên mã cổ phiếu"""
    url = API_BASE_URL.format(symbol=symbol)
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data)
        if not df.empty:
            df["date"] = df["date"].astype(str)  
        return df
    else:
        st.error(f"❌ Lỗi lấy dữ liệu từ API: {response.status_code}")
        return pd.DataFrame()

    st.title("📊 Thanh khoản thị trường (Nguồn: CafeF)")

    symbol = st.text_input("Nhập mã cổ phiếu (VD: REE, MWG, HPG):").upper()

    if symbol:
        df = fetch_liquidity_data(symbol)

        if not df.empty:
            df = df[["date", "gtgD1", "gtgD2"]]  

            tick_indices = list(range(0, len(df), max(len(df) // 10, 1)))
            tick_labels = [df["date"].iloc[i] for i in tick_indices]

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=df["date"], y=df["gtgD1"], 
                mode="lines", name="GTGD phiên hiện tại", 
                line=dict(color="orange", width=3)
            ))
            
            fig.add_trace(go.Scatter(
                x=df["date"], y=df["gtgD2"], 
                mode="lines", name="GTGD phiên trước", 
                line=dict(color="gray", width=3)
            ))

            fig.update_layout(
                title=f"📈 Thanh khoản thị trường - {symbol}",
                xaxis_title="Thời gian",
                yaxis_title="Giá trị Giao dịch (Tỷ đồng)",
                xaxis=dict(
                    type="category",
                    tickmode="array",
                    tickvals=tick_labels,
                    ticktext=tick_labels
                ),
                yaxis=dict(
                    tickformat=",.0f"  
                ),
                legend=dict(x=0, y=1)
            )

            st.plotly_chart(fig)

        else:
            st.warning("⚠ Không có dữ liệu!")

import streamlit as st
from pers_teslim.py import Trader, Long_Short_Backtester  # your_module'un yerini kodunuzun bulunduğu dosyanın adı ile değiştirin

def main():
    st.title('Crypto Trading Bot')
    
    # Kullanıcıdan giriş bilgilerini alalım
    api_key = st.text_input("aF7Sq8sM6yJfVe7sN2XjDzxLuZt45mybHOPrEvfAAahqA2hQvoElA8Bo7v8LJe08")
    api_secret = st.text_input("RJSWSD29RszE3BMJ1xi4f2i11OXGc4Igcz7BbQcjB9PFFBSHwng4qVtXMULeltfd")
    
    # Trader objesi oluşturalım
    client = Client(api_key, api_secret)
    trader = Trader(client)
    
    # Kullanıcıdan hangi kripto parayla işlem yapmak istediğini soralım
    symbol = st.text_input("Symbol", "BTCUSDT")
    
    # Trader'ın verilerini alalım ve bu veriyi dataframe olarak gösterelim
    trader.get_data(symbol=symbol)
    st.dataframe(trader.df)
    
    # Backtester objesi oluşturalım
    filepath = st.text_input("bitcoin.csv")
    backtester = Long_Short_Backtester(filepath, symbol, start, end, tc)
    
    # Backtester'ın verilerini alalım ve bu veriyi dataframe olarak gösterelim
    backtester.get_data()
    st.dataframe(backtester.data)
    
    # Stratejiyi test edelim ve sonuçları gösterelim
    smas = st.text_input("Enter SMA values (comma-separated)", "10,20,30")
    smas = [int(sma) for sma in smas.split(",")]
    backtester.test_strategy(smas)
    st.write(f"Performance: {backtester.perf}")
    st.line_chart(backtester.data[['price', 'SMA_S', 'SMA_M', 'SMA_L']])
    
if __name__ == "__main__":
    main()


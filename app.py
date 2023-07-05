import streamlit as st
from pers_teslim.py import Trader, Long_Short_Backtester  # your_module'un yerini kodunuzun bulunduğu dosyanın adı ile değiştirin

def main():
    st.title('Crypto Trading Bot')
    
    # Kullanıcıdan giriş bilgilerini alalım
    api_key = st.text_input("rDy56ksG5yMauFSNUkX3Bklg4GMjGQ0gvqTCUyi5KYxMzYHkl0HtA0XvTCpJxZQC")
    api_secret = st.text_input("incVE8AX8aGyIj21UDKEjdJuYCbNFk8EN7x4FMD9aKKClQhMzm2dJ3uFx3mjgYhI")
    
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


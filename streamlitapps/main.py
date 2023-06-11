import streamlit as st
import home_page, health_query

PAGES = {
    "Home": home_page,
    "Healthcare Query": health_query
}

def main():
    st.sidebar.title('Navigation')
    choice = st.sidebar.radio("Go to", list(PAGES.keys()))

    PAGES[choice].app()

if __name__ == "__main__":
    main()

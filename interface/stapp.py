import streamlit as st

# Initialize session state for conversation log and turn
if 'conversation' not in st.session_state:
    st.session_state.conversation = []
if 'turn' not in st.session_state:
    st.session_state.turn = 0
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []

# Function to add a message to the conversation log
def add_message(speaker, message):
    st.session_state.conversation.append({"speaker": speaker, "message": message})

# Function to handle the next turn
def handle_next_turn():
    agents = ["Your Lawyer", "Opposition Lawyer", "Judge"]
    messages = {
        "Your Lawyer": "Your lawyer is presenting the case...",
        "Opposition Lawyer": "Opposition lawyer is responding...",
        "Judge": "Judge is reviewing the arguments..."
    }
    
    current_agent = agents[st.session_state.turn % len(agents)]
    add_message(current_agent, messages[current_agent])
    st.session_state.turn += 1

# Set the page configuration
st.set_page_config(
    page_title="Courtroom Simulator",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Title of the App
st.title("🏛️ Courtroom Simulator")

# Description
st.markdown("""
Welcome to the **Courtroom Simulator**! Upload your case documents and watch as the courtroom unfolds with interactions between **Your Lawyer**, the **Opposition Lawyer**, and the **Judge**.
""")

# Sidebar for File Upload
st.sidebar.header("📂 Upload Documents")
uploaded_files = st.sidebar.file_uploader(
    "Choose case documents",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True
)

if uploaded_files:
    st.session_state.uploaded_files = uploaded_files
    st.sidebar.success(f"{len(uploaded_files)} file(s) uploaded.")
    for file in uploaded_files:
        st.sidebar.markdown(f"- 📄 {file.name}")
else:
    st.sidebar.info("No documents uploaded yet.")

st.markdown("---")

# Main Area
col1, col2 = st.columns([3, 1])

with col1:
    st.header("🗨️ Courtroom Proceedings")
    
    # Display the conversation log
    for entry in st.session_state.conversation:
        if entry["speaker"] == "Judge":
            st.markdown(f"**👩‍⚖️ {entry['speaker']}:** {entry['message']}")
        elif entry["speaker"] == "Your Lawyer":
            st.markdown(f"**🧑‍⚖️ {entry['speaker']}:** {entry['message']}")
        elif entry["speaker"] == "Opposition Lawyer":
            st.markdown(f"**👨‍⚖️ {entry['speaker']}:** {entry['message']}")
    
    # Button to proceed to the next turn
    if st.button("▶️ Next Turn"):
        handle_next_turn()

with col2:
    st.header("🔍 Case Details")
    
    if st.session_state.uploaded_files:
        st.subheader("📄 Uploaded Documents")
        for file in st.session_state.uploaded_files:
            st.markdown(f"- {file.name}")
            # Optionally, provide download links
            st.download_button(
                label="Download",
                data=file,
                file_name=file.name,
                mime='application/octet-stream'
            )
    else:
        st.info("No documents uploaded.")

    st.markdown("---")
    
    st.subheader("🎯 Current Turn")
    agents = ["Your Lawyer", "Opposition Lawyer", "Judge"]
    current_agent = agents[st.session_state.turn % len(agents)]
    st.markdown(f"**Next Speaker:** {current_agent}")

# Footer
st.markdown("""
---
*Courtroom Simulator App - Powered by Streamlit*
""")


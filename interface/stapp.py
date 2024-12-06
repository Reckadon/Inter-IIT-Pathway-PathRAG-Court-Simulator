import asyncio
import aiohttp
import streamlit as st
import json
import os
from pathlib import Path

# Create private_documents directory if it doesn't exist
UPLOAD_DIR = Path("private_documents")
UPLOAD_DIR.mkdir(exist_ok=True)

async def fetch_stream(user_prompt):
    url = "http://localhost:8000/stream_workflow"
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json={"user_prompt": user_prompt}) as response:
            if response.status != 200:
                st.error(f"Failed to connect: {response.status}")
                return
            async for line in response.content:
                if line:
                    yield line.decode("utf-8")

st.title("🏛️ Courtroom Simulator")

# Add file uploader before the text input
uploaded_files = st.file_uploader(
    "Upload case-related documents",
    accept_multiple_files=True,
    type=['pdf', 'txt', 'doc', 'docx']
)

# Handle file uploads
if uploaded_files:
    for uploaded_file in uploaded_files:
        # Create safe filename
        file_path = UPLOAD_DIR / uploaded_file.name
        
        # Save the file
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"Saved: {uploaded_file.name}")

user_prompt = st.text_input("Enter your case details:", """Case File

            Case Title
            State vs. Rohan Malhotra

            Case Summary
            Rohan Malhotra, a 28-year-old entrepreneur, is accused of defamation under Section 499 of the Indian Penal Code (IPC) and cyber harassment under the Information Technology Act, 2000. The case pertains to Rohan allegedly posting defamatory and harassing statements about Meera Sharma, a 30-year-old journalist, on social media platforms. Rohan denies the charges, claiming his account was hacked at the time the posts were made.

            Case Details
            Incident Description:
            On May 20, 2024, Meera Sharma filed a complaint with the cybercrime division, alleging that Rohan Malhotra made a series of defamatory posts about her on Twitter and Instagram. The posts accused Meera of biased reporting, bribery, and professional misconduct, damaging her reputation among peers and the public.
            Rohan Malhotra has stated that he did not make the posts and believes his accounts were compromised. He claims that he noticed suspicious activity on his accounts around the time the posts were made.
            Timeline of Events:
            May 19, 2024, 8:00 PM: First defamatory tweet posted.
            May 19, 2024, 8:30 PM: Instagram story containing similar allegations shared.
            May 20, 2024, 9:00 AM: Meera Sharma files a complaint.
            May 20, 2024, 11:00 AM: Rohan reports to the cybercrime division, claiming his accounts were hacked.
            Evidence:
            Screenshots of Posts: Captured by Meera Sharma before they were deleted.
            Digital Forensics Report: Analysis of Rohan's devices shows no direct evidence of his involvement but logs indicate suspicious login activity from an unknown IP address.
            Witness Testimony: Ravi Verma, a friend of Rohan, states that Rohan mentioned concerns about his social media accounts prior to the incident.
            Impact Statement: Meera Sharma has submitted a statement detailing the professional and emotional impact caused by the defamatory posts.
            Charges:
            Section 499, IPC (Defamation):
            Whoever, by words, either spoken or intended to be read, or by signs or visible representations, makes or publishes any imputation concerning any person intending to harm, or knowing or having reason to believe that such imputation will harm, the reputation of such person, is said to defame that person.


            Section 66A, IT Act (Cyber Harassment):
            Punishment for sending offensive messages through communication service, etc., which cause annoyance or insult.
            """
    )

if st.button("Run Workflow"):
    st.write("Streaming results:")
    placeholder = st.empty()

    async def stream_workflow():
        try:
            async for event in fetch_stream(user_prompt):
                if event.startswith("data: "):  # SSE events start with `data: `
                    raw_data = event[6:].strip()  # Remove the `data: ` prefix
                    st.write(f"Raw Event: {raw_data}")
                    try:
                        parsed_data = json.loads(raw_data)
                        placeholder.write(parsed_data)  # Display parsed JSON
                    except json.JSONDecodeError as e:
                        st.error(f"Error decoding event data: {e}")
        except Exception as e:
            st.error(f"Error streaming workflow: {e}")

    asyncio.run(stream_workflow())

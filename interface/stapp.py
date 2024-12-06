import asyncio
import aiohttp
import streamlit as st
import json

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

user_prompt = st.text_input("Enter your case details:", "State vs Alex Martin")

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

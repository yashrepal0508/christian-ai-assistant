import os
import streamlit as st
from dotenv import load_dotenv

from core.llm import get_llm_response
from core.bible_rag import retrieve_relevant_verses, extract_and_verify_citations
from core.moderation import pre_filter, classify_image_prompt, check_response_safety
from core.image_gen import generate_christian_image
from core.memory import trim_history

load_dotenv()

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Christian AI Assistant",
    page_icon="✝",
    layout="centered",
)

st.title("✝ Christian AI Assistant")
st.caption("Grounded in Scripture • Denomination-aware • Safe")

# ── Session state init ─────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "denomination" not in st.session_state:
    st.session_state.denomination = "General Christian (non-denominational)"
if "mode" not in st.session_state:
    st.session_state.mode = "Chat"

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")

    st.session_state.denomination = st.selectbox(
        "Your denomination",
        [
            "General Christian (non-denominational)",
            "Protestant (Evangelical)",
            "Protestant (Reformed / Calvinist)",
            "Catholic",
            "Eastern Orthodox",
            "Lutheran",
            "Methodist",
            "Baptist",
            "Pentecostal / Charismatic",
            "Anglican / Episcopal",
        ],
        index=0,
    )

    st.session_state.mode = st.radio("Mode", ["Chat", "Image Generation"])

    st.markdown("---")
    st.markdown("**About**")
    st.markdown(
        "This assistant uses:\n"
        "- Mistral-7B (HF)\n"
        "- Bible RAG (ChromaDB)\n"
        "- Verse verification (bible-api.com)\n"
        "- SDXL image gen (HF)"
    )

    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()


# ── Display chat history ───────────────────────────────────────────────────────
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    elif msg["role"] == "assistant":
        with st.chat_message("assistant", avatar="✝"):
            st.markdown(msg["content"])
    elif msg["role"] == "image":
        with st.chat_message("assistant", avatar="✝"):
            st.image(msg["image"], caption=msg["caption"])


# ── Image generation mode ──────────────────────────────────────────────────────
if st.session_state.mode == "Image Generation":
    st.info("Describe a Christian scene or theme to generate an image.")
    img_prompt = st.chat_input("Describe a Christian image (e.g. 'The Good Shepherd')")

    if img_prompt:
        with st.chat_message("user"):
            st.markdown(img_prompt)
        st.session_state.messages.append({"role": "user", "content": img_prompt})

        safety = classify_image_prompt(img_prompt)
        if not safety["safe"]:
            with st.chat_message("assistant", avatar="✝"):
                st.error(safety["reason"])
            st.session_state.messages.append(
                {"role": "assistant", "content": f"**[Blocked]** {safety['reason']}"}
            )
        else:
            with st.spinner("Generating Christian artwork..."):
                image = generate_christian_image(safety["refined_prompt"])
            if image:
                with st.chat_message("assistant", avatar="✝"):
                    st.image(image, caption=f'"{img_prompt}" — Christian art')
                st.session_state.messages.append(
                    {"role": "image", "image": image, "caption": f'"{img_prompt}"'}
                )
            else:
                with st.chat_message("assistant", avatar="✝"):
                    st.warning("Image generation failed. Please try again or check your HF token.")

# ── Chat mode ──────────────────────────────────────────────────────────────────
else:
    user_input = st.chat_input("Ask about scripture, theology, Christian living...")

    if user_input:
        # Show user message immediately
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Stage 1: Pre-filter moderation
        mod_result = pre_filter(user_input)

        if mod_result["blocked"]:
            response_text = f"I'm not able to help with that request. {mod_result['reason']}"
            with st.chat_message("assistant", avatar="✝"):
                st.error(response_text)
            st.session_state.messages.append({"role": "assistant", "content": response_text})

        else:
            if mod_result["warn"]:
                st.info(f"Note: {mod_result['reason']}")

            # Stage 2: RAG — retrieve relevant Bible passages
            with st.spinner("Searching scripture..."):
                context = retrieve_relevant_verses(user_input, n_results=4)

            # Stage 3: LLM response
            with st.spinner("Reflecting on scripture..."):
                chat_history = trim_history(
                    [m for m in st.session_state.messages if m["role"] in ("user", "assistant")]
                )
                response_text = get_llm_response(
                    messages=chat_history,
                    denomination=st.session_state.denomination,
                    context=context,
                )

            # Stage 4: Post-response safety check
            safety_check = check_response_safety(response_text)
            if not safety_check["safe"]:
                response_text = (
                    "I generated a response but it flagged our safety check. "
                    "Please rephrase your question."
                )

            # Display response
            with st.chat_message("assistant", avatar="✝"):
                st.markdown(response_text)

                # Show retrieved context in expander
                if context:
                    with st.expander("Scripture passages used for grounding"):
                        st.markdown(context)

                # Verify any citations made in the response
                citations = extract_and_verify_citations(response_text)
                if citations:
                    with st.expander("Verse verification"):
                        for c in citations:
                            if c["found"]:
                                st.success(f"**{c['reference']}** — verified ✓")
                                st.caption(c["text"])
                            else:
                                st.warning(f"**{c['reference']}** — could not verify against KJV")

            st.session_state.messages.append({"role": "assistant", "content": response_text})

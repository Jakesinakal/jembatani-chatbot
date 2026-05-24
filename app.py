import os
import streamlit as st
from google import genai
from google.genai import types
from dotenv import load_dotenv
from tools import get_harga_sayuran
from rag_setup import setup_rag, query_rag

load_dotenv()

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Jembatani Chatbot",
    page_icon="🌾",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🌾 Jembatani")
    st.markdown("*Menghubungkan Petani & Konsumen Indonesia*")
    st.divider()
    st.markdown("**Yang bisa saya bantu:**")
    st.markdown("- 💰 Cek harga sayuran per daerah")
    st.markdown("- 📖 Panduan penggunaan aplikasi")
    st.markdown("- 🌿 Informasi seputar pertanian")
    st.divider()
    st.markdown("**Daerah yang tersedia:**")
    st.markdown("Jakarta · Bandung · Surabaya\nYogyakarta · Medan · Makassar")
    st.divider()
    if st.button("🗑️ Hapus Riwayat Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.history = []
        st.rerun()

# ---------------------------------------------------------------------------
# Inisialisasi Gemini client (SDK baru: google.genai)
# ---------------------------------------------------------------------------
@st.cache_resource
def init_client():
    return genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

client = init_client()

# ---------------------------------------------------------------------------
# Inisialisasi RAG
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Memuat knowledge base panduan aplikasi...")
def init_rag():
    return setup_rag()

vectorstore = init_rag()

# ---------------------------------------------------------------------------
# System prompt persona Jembatani
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """
Kamu adalah Jembatani Chatbot, asisten virtual ramah dari aplikasi Jembatani —
platform digital yang menghubungkan petani langsung dengan konsumen di seluruh Indonesia.

Tugasmu:
1. Membantu pengguna (petani atau konsumen) mendapatkan informasi harga sayuran segar di berbagai daerah.
2. Menjelaskan cara penggunaan dan mekanisme aplikasi Jembatani.
3. Menjawab pertanyaan umum seputar pertanian dengan ramah.

Panduan komunikasi:
- Gunakan Bahasa Indonesia yang ramah, sopan, dan mudah dipahami semua kalangan.
- Sapa pengguna dengan hangat di awal percakapan.
- Jika ditanya harga sayuran, SELALU gunakan tool get_harga_sayuran untuk mendapatkan data.
- Jika ada pertanyaan tentang cara penggunaan aplikasi, jawab berdasarkan konteks panduan yang diberikan.
- Jawab dengan singkat, jelas, dan informatif.
- Jika informasi tidak tersedia, sampaikan dengan jujur dan tawarkan bantuan lain.
"""

# ---------------------------------------------------------------------------
# Tool definition untuk function calling
# ---------------------------------------------------------------------------
harga_tool = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="get_harga_sayuran",
            description="Mendapatkan harga sayuran berdasarkan nama sayuran dan nama daerah/kota.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "nama_sayur": types.Schema(
                        type="STRING",
                        description="Nama sayuran (contoh: Cabai Merah, Tomat, Bawang Merah, Bayam)",
                    ),
                    "daerah": types.Schema(
                        type="STRING",
                        description="Nama daerah atau kota (contoh: Jakarta, Bandung, Surabaya, Yogyakarta, Medan, Makassar)",
                    ),
                },
                required=["nama_sayur", "daerah"],
            ),
        )
    ]
)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "history" not in st.session_state:
    # History dalam format google.genai types.Content untuk multi-turn chat
    st.session_state.history = []

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🌾 Jembatani Chatbot")
st.caption("Asisten virtual untuk petani dan konsumen Indonesia")

# ---------------------------------------------------------------------------
# Tampilkan riwayat chat
# ---------------------------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Pesan sambutan jika chat masih kosong
if not st.session_state.messages:
    with st.chat_message("assistant"):
        welcome = (
            "Halo! 👋 Selamat datang di **Jembatani Chatbot**!\n\n"
            "Saya siap membantu kamu dengan:\n"
            "- 💰 **Cek harga sayuran** di berbagai daerah "
            "(Jakarta, Bandung, Surabaya, Yogyakarta, Medan, Makassar)\n"
            "- 📖 **Panduan aplikasi** Jembatani "
            "(cara daftar, berjualan, berbelanja, pembayaran, dll)\n"
            "- 🌿 **Informasi pertanian** umum\n\n"
            "Ada yang bisa saya bantu hari ini? 😊"
        )
        st.markdown(welcome)

# ---------------------------------------------------------------------------
# Helper: jalankan function calling jika Gemini memintanya
# ---------------------------------------------------------------------------
def execute_function_calls(response) -> list[types.Part] | None:
    """Periksa apakah ada function call di response, lalu jalankan."""
    parts = []
    for candidate in response.candidates:
        for part in candidate.content.parts:
            if part.function_call:
                fc = part.function_call
                if fc.name == "get_harga_sayuran":
                    result = get_harga_sayuran(**dict(fc.args))
                else:
                    result = "Fungsi tidak dikenali."

                parts.append(
                    types.Part.from_function_response(
                        name=fc.name,
                        response={"result": result},
                    )
                )
    return parts if parts else None

# ---------------------------------------------------------------------------
# Input pengguna
# ---------------------------------------------------------------------------
if prompt := st.chat_input("Ketik pertanyaan kamu di sini..."):

    # Ambil konteks RAG yang relevan dari panduan aplikasi
    rag_context = query_rag(vectorstore, prompt)

    # Inject konteks RAG ke dalam pesan (manual RAG augmentation)
    if rag_context:
        augmented_prompt = (
            "[Konteks dari panduan aplikasi Jembatani — gunakan jika relevan]\n"
            f"{rag_context}\n"
            "[/Konteks]\n\n"
            f"Pertanyaan pengguna: {prompt}"
        )
    else:
        augmented_prompt = prompt

    # Tampilkan pesan asli user (bukan augmented)
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Tambahkan ke history multi-turn
    st.session_state.history.append(
        types.Content(role="user", parts=[types.Part.from_text(text=augmented_prompt)])
    )

    # Dapatkan respons dari Gemini
    with st.chat_message("assistant"):
        with st.spinner("Sedang memproses..."):
            try:
                config = types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    tools=[harga_tool],
                    temperature=0.7,
                )

                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=st.session_state.history,
                    config=config,
                )

                # Handle function calling (maksimal 5 putaran)
                for _ in range(5):
                    fn_parts = execute_function_calls(response)
                    if not fn_parts:
                        break

                    # Simpan respons model (yang berisi function call) ke history
                    st.session_state.history.append(response.candidates[0].content)

                    # Kirim hasil function ke Gemini
                    st.session_state.history.append(
                        types.Content(role="user", parts=fn_parts)
                    )
                    response = client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=st.session_state.history,
                        config=config,
                    )

                final_text = response.text

                # Simpan respons akhir ke history
                st.session_state.history.append(response.candidates[0].content)

                st.markdown(final_text)
                st.session_state.messages.append({"role": "assistant", "content": final_text})

            except Exception as e:
                st.error(f"Maaf, terjadi kesalahan: {str(e)}")

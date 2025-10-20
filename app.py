import time
import numpy as np
import cv2
import streamlit as st
import pandas as pd
import sqlite3
from datetime import datetime
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration, VideoTransformerBase
from tensorflow.keras.models import load_model
import plotly.express as px
import asyncio

# --- Configuración de la página ---
st.set_page_config(page_title="Clasificador en vivo", page_icon="🎥", layout="wide")
st.title("🎥 Clasificación en vivo con Keras + Streamlit")
st.caption("Cámara dentro de la página y resultados en la misma interfaz. Incluye selector de cámara/calidad, registro a CSV y SQLite.")

MODEL_PATH = "keras_Model.h5"
LABELS_PATH = "labels.txt"
DB_PATH = "predicciones.db"

# --- Funciones de carga ---
@st.cache_resource
def load_model_cached(model_path: str):
    return load_model(model_path, compile=False)

@st.cache_data
def load_labels(labels_path: str):
    with open(labels_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines()]

try:
    model = load_model_cached(MODEL_PATH)
    labels = load_labels(LABELS_PATH)
except Exception as e:
    st.error(f"No se pudo cargar el modelo/etiquetas: {e}")
    st.stop()

# --- Configuración SQLite ---
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
c = conn.cursor()
c.execute("""
CREATE TABLE IF NOT EXISTS predicciones (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    clase TEXT,
    confianza REAL
)
""")
conn.commit()

def save_prediction_db(clase, confianza):
    c.execute("INSERT INTO predicciones (timestamp, clase, confianza) VALUES (?, ?, ?)",
              (datetime.utcnow().isoformat(), clase, confianza))
    conn.commit()

# --- Tabla de personas ---
c.execute("""
CREATE TABLE IF NOT EXISTS personas (
    clase TEXT PRIMARY KEY,
    nombre TEXT,
    edad INTEGER,
    email TEXT,
    telefono TEXT,
    foto_url TEXT
)
""")

personas = [
    ("wendy", "Wendy Nicole Llivichuzhca", 20, "wendy@gmail.com", "0999999999", "https://i.vgy.me/R3EGsB.jpg"),
    ("adriana", "Adriana Valentina Cornejo Ulloa", 20, "adriana@gmail.com", "0998887777", "https://i.vgy.me/nsc6vR.jpg")
]

for p in personas:
    c.execute("""
        INSERT OR IGNORE INTO personas (clase, nombre, edad, email, telefono, foto_url) 
        VALUES (?, ?, ?, ?, ?, ?)""", p)
conn.commit()

# --- Tarjeta por defecto para desconocidos ---
DESCONOCIDO = {
    "nombre": "Desconocido",
    "edad": None,
    "email": None,
    "telefono": None,
    "foto_url": "https://i.vgy.me/FqhKAk.jpg"
}

# --- Función para mostrar tarjeta ---
def mostrar_tarjeta_persona(clase_detectada, placeholder):
    # Limpiar placeholder antes de mostrar nueva tarjeta
    placeholder.empty()

    # Determinar si es desconocido
    if not clase_detectada or clase_detectada.lower() == "desconocido":
        fila = None
    else:
        clase_clean = "".join([c for c in clase_detectada if c.isalpha()]).lower()
        c.execute("SELECT * FROM personas WHERE LOWER(clase) = ?", (clase_clean,))
        fila = c.fetchone()

    # Crear nuevo contenido
    with placeholder.container():
        col1, col2 = st.columns([1,2])
        if fila:
            foto_url = fila[5] if fila[5] else DESCONOCIDO["foto_url"]
            with col1:
                st.image(foto_url, width=120)
            with col2:
                st.markdown(f"**Nombre:** {fila[1]}")
                st.markdown(f"**Edad:** {fila[2]}")
                st.markdown(f"**Email:** [{fila[3]}](mailto:{fila[3]})")
                st.markdown(f"**Teléfono:** {fila[4]}")
        else:
            with col1:
                st.image(DESCONOCIDO["foto_url"], width=120)
            with col2:
                st.markdown("**Persona desconocida**")
                st.markdown("No hay información disponible")

# --- Sidebar ---
st.sidebar.header("Ajustes de cámara")
facing = st.sidebar.selectbox("Tipo de cámara (facingMode)", ["auto (por defecto)", "user (frontal)", "environment (trasera)"], index=0)
quality = st.sidebar.selectbox("Calidad de video", ["640x480", "1280x720", "1920x1080"], index=1)
w, h = map(int, quality.split("x"))
video_constraints = {"width": w, "height": h}
if facing != "auto (por defecto)":
    video_constraints["facingMode"] = facing.split(" ")[0]

media_constraints = {"video": video_constraints, "audio": False}

st.sidebar.header("Registro de predicciones")
enable_log = st.sidebar.checkbox("Habilitar registro (SQLite + CSV)", value=True)
log_every_n_seconds = st.sidebar.slider("Intervalo de registro (s)", 0.2, 5.0, 1.0, 0.2)

if "last_log_ts" not in st.session_state:
    st.session_state.last_log_ts = 0.0

# --- WebRTC config ---
RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

class VideoTransformer(VideoTransformerBase):
    def __init__(self) -> None:
        self.latest = {"class": None, "confidence": 0.0}
        self.model = model
        self.labels = labels

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        resized = cv2.resize(img, (224,224), interpolation=cv2.INTER_AREA)
        x = resized.astype(np.float32).reshape(1,224,224,3)
        x = (x / 127.5) - 1.0
        pred = self.model.predict(x, verbose=0)
        idx = int(np.argmax(pred))
        label = self.labels[idx] if idx < len(self.labels) else f"Clase {idx}"
        conf = float(pred[0][idx])
        self.latest = {"class": label, "confidence": conf}

        # Overlay con detección desconocido
        display_label = label if conf >= 0.5 else "desconocido"
        overlay = img.copy()
        text = f"{display_label} | {conf*100:.1f}%"
        cv2.rectangle(overlay, (5,5), (5+8*len(text), 45), (0,0,0), -1)
        cv2.putText(overlay, text, (10,35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2, cv2.LINE_AA)
        return overlay

# --- Layout ---
left, right = st.columns([2,1], gap="large")

with left:
    st.subheader("Cámara en vivo")
    webrtc_ctx = webrtc_streamer(
        key="keras-live",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints=media_constraints,
        video_transformer_factory=VideoTransformer,
        async_processing=True
    )

with right:
    st.subheader("Resultados en tiempo real")
    result_placeholder = st.empty()
    progress_placeholder = st.empty()
    persona_placeholder = st.empty()
    st.write("Etiqueta detectada (debug):")  # DEBUG opcional

# --- Loop de predicciones ---
async def actualizar_predicciones(webrtc_ctx, result_placeholder, progress_placeholder, persona_placeholder):
    UMBRAL_DESCONOCIDO = 0.5
    while webrtc_ctx.state.playing:
        vt = webrtc_ctx.video_transformer
        if vt and vt.latest["class"]:
            raw_cls = vt.latest["class"]
            conf = vt.latest["confidence"]

            if conf < UMBRAL_DESCONOCIDO:
                cls = "desconocido"
                raw_cls = "desconocido"
            else:
                cls = raw_cls

            # Mostrar resultados
            result_placeholder.markdown(f"**Clase detectada:** `{cls}`\n**Confianza:** `{conf*100:.2f}%`")
            progress_placeholder.progress(min(max(conf,0.0),1.0))

            # Mostrar tarjeta persona limpiando primero
            mostrar_tarjeta_persona(raw_cls, persona_placeholder)

            # Guardar en DB si habilitado
            if enable_log:
                now = time.time()
                if now - st.session_state.last_log_ts >= log_every_n_seconds:
                    save_prediction_db(cls, round(conf,6))
                    st.session_state.last_log_ts = now

        await asyncio.sleep(0.2)

if webrtc_ctx and webrtc_ctx.state.playing:
    asyncio.run(actualizar_predicciones(webrtc_ctx, result_placeholder, progress_placeholder, persona_placeholder))

# --- Modo alternativo con foto ---
st.markdown("---")
with st.expander("⚠️ Modo alternativo (captura por foto)"):
    snap = st.camera_input("Captura una imagen")
    if snap is not None:
        file_bytes = np.asarray(bytearray(snap.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        resized = cv2.resize(img, (224,224), interpolation=cv2.INTER_AREA)
        x = resized.astype(np.float32).reshape(1,224,224,3)
        x = (x / 127.5) - 1.0
        pred = model.predict(x, verbose=0)
        idx = int(np.argmax(pred))
        label = labels[idx] if idx < len(labels) else f"Clase {idx}"
        conf = float(pred[0][idx])

        if conf < 0.5:
            label = "desconocido"

        st.image(img, caption=f"{label} | {conf*100:.2f}%")
        st.success(f"Predicción: **{label}** ({conf*100:.2f}%)")
        if enable_log:
            save_prediction_db(label, round(conf,6))
            mostrar_tarjeta_persona(label, persona_placeholder)

# --- Dashboard Interactivo con Plotly ---
st.markdown("---")
st.subheader("📊 Panel Analítico")

df = pd.read_sql_query("SELECT * FROM predicciones", conn)

if not df.empty:
    df['hora'] = pd.to_datetime(df['timestamp']).dt.hour

    # Distribución de etiquetas (Dona)
    etiqueta_counts = df['clase'].value_counts().reset_index()
    etiqueta_counts.columns = ['etiqueta', 'conteo']
    fig1 = px.pie(etiqueta_counts, values='conteo', names='etiqueta', hole=0.4,
                  title="Distribución de etiquetas", color_discrete_sequence=px.colors.qualitative.Set2)
    st.plotly_chart(fig1, use_container_width=True)

    # Confianza por etiqueta (Bubble)
    df['confianza_scaled'] = df['confianza']*100
    fig2 = px.scatter(df, x='clase', y='confianza', size='confianza_scaled',
                      color='clase', hover_data=['timestamp'],
                      title="Confianza por etiqueta (Bubble Chart)", size_max=60,
                      color_discrete_sequence=px.colors.qualitative.Set1)
    st.plotly_chart(fig2, use_container_width=True)

    # Predicciones por hora (Barras)
    hora_counts = df['hora'].value_counts().sort_index().reset_index()
    hora_counts.columns = ['hora', 'count']
    fig3 = px.bar(hora_counts, x='hora', y='count',
                  labels={'hora':'Hora', 'count':'Cantidad'},
                  title="Predicciones por hora",
                  text='count',
                  color_discrete_sequence=px.colors.qualitative.Pastel)
    st.plotly_chart(fig3, use_container_width=True)

    # Confianza promedio por etiqueta (Barras)
    df_mean = df.groupby('clase')['confianza'].mean().reset_index()
    fig4 = px.bar(df_mean, x='clase', y='confianza', color='clase',
                  title="Confianza promedio por etiqueta", text=df_mean['confianza'],
                  color_discrete_sequence=px.colors.qualitative.Set2)
    fig4.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    st.plotly_chart(fig4, use_container_width=True)

    # Confianza vs Hora por clase (Bubble)
    fig5 = px.scatter(df, x='hora', y='confianza', size='confianza_scaled', color='clase',
                      hover_data=['timestamp'], size_max=50,
                      title="Confianza por hora y clase")
    st.plotly_chart(fig5, use_container_width=True)

    # Conteo acumulado por clase (línea)
    df_line = df.groupby(pd.to_datetime(df['timestamp']).dt.date)['clase'].value_counts().reset_index(name='count')
    fig6 = px.line(df_line, x='timestamp', y='count', color='clase',
                   title="Conteo de predicciones por día y clase")
    st.plotly_chart(fig6, use_container_width=True)

    # Descargar CSV actualizado
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Descargar CSV de predicciones", data=csv_bytes,
                       file_name=f"predicciones_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                       mime="text/csv",
                       disabled=df.empty)
else:
    st.info("No hay datos para mostrar en el panel analítico aún.")

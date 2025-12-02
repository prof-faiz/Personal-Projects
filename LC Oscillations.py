import streamlit as st
import numpy as np
from scipy.integrate import solve_ivp
import plotly.graph_objs as go
import time


# -----------------------------------------------------------
# ANALYTICAL LC SOLVER (dimensionless time)
# -----------------------------------------------------------
def analytical_LC(tau):
    q = np.cos(tau)
    i = -np.sin(tau)
    return q, i


# -----------------------------------------------------------
# NUMERICAL (still solves physical) but returns normalized
# -----------------------------------------------------------
@st.cache_data
def numerical_LC(t_start, t_end, L, C, Q0=1.0, I0=0.0):
    def f(t, y):
        q, i = y
        return [i, -(1/(L*C)) * q]

    t_eval = np.linspace(t_start, t_end, 20000)
    sol = solve_ivp(f, (t_start, t_end), [Q0, I0], t_eval=t_eval)

    # normalize physical time to tau = omega t
    omega = 1/np.sqrt(L*C)
    tau = omega * sol.t

    return tau, sol.y[0], sol.y[1]


# -----------------------------------------------------------
# PLOT WINDOW
# -----------------------------------------------------------
def plot_window(tau, q, i, tau_cut, window):
    tau_min = tau_cut - window
    mask = (tau >= tau_min) & (tau <= tau_cut)

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=tau[mask], y=q[mask],
                             mode='lines', name='Charge q'))
    fig.add_trace(go.Scatter(x=tau[mask], y=i[mask],
                             mode='lines', name='Current i'))

    fig.update_layout(
        xaxis=dict(range=[tau_min, tau_cut], title="Normalized Time τ"),
        yaxis=dict(range=[-1.1, 1.1], title="Amplitude"),
        height=450,
        margin=dict(l=10, r=10, t=10, b=30),
        legend=dict(orientation="h", y=1.1)
    )
    return fig


# -----------------------------------------------------------
# STREAMLIT UI
# -----------------------------------------------------------
st.title("LC Oscillation Simulator")

logL = st.slider("log10(L) [H]", -6.0, -2.0, -4.0)
logC = st.slider("log10(C) [F]", -12.0, -6.0, -9.0)

L = 10**logL
C = 10**logC

mode = st.radio("Mode", ["Analytical", "Numerical"], horizontal=True)

# animation rate
speed = st.slider("Animation speed", 0.05, 1.0, 0.3)

# normalized domain (~100 cycles)
tau_end = 200*np.pi
window = 6*np.pi       # ~3 cycles visible
dt = 0.04 * speed     # animation increment


# -----------------------------------------------------------
# SOLVE
# -----------------------------------------------------------
if mode == "Analytical":
    tau = np.linspace(0, tau_end, 20000)
    q, i = analytical_LC(tau)
else:
    tau, q, i = numerical_LC(0, tau_end, L, C)


# session state
if "tau_pos" not in st.session_state:
    st.session_state.tau_pos = 0.0
if "playing" not in st.session_state:
    st.session_state.playing = False


col1, col2 = st.columns(2)
if col1.button("▶ Play"):
    st.session_state.playing = True
if col2.button("⏸ Pause"):
    st.session_state.playing = False


plot_area = st.empty()


# -----------------------------------------------------------
# ANIMATION LOOP
# -----------------------------------------------------------
if st.session_state.playing:

    for _ in range(200):
        if not st.session_state.playing:
            break

        st.session_state.tau_pos += dt
        if st.session_state.tau_pos > tau_end:
            st.session_state.tau_pos = 0.0

        fig = plot_window(tau, q, i, st.session_state.tau_pos, window)
        plot_area.plotly_chart(fig, use_container_width=True)

        time.sleep(0.03)

else:
    fig = plot_window(tau, q, i, st.session_state.tau_pos, window)
    plot_area.plotly_chart(fig, use_container_width=True)

"""
OUROBOROS Interdisciplinary Visualizations
===========================================
Visualization functions for algorithms 11-20
Generates physics-inspired, biology-inspired, and audio-inspired visuals

Author: OUROBOROS Project
Date: December 7, 2025
"""

import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from plotly.subplots import make_subplots


# ═══════════════════════════════════════════════════════════════════════════
# THEME-AWARE PLOTLY CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════
def get_plotly_theme_config():
    """
    Returns theme-aware Plotly layout configuration.
    Uses light theme for all visualizations.
    """
    return {
        'template': 'plotly_white',  # Light theme
        'font': {
            'family': 'JetBrains Mono, monospace',
            'size': 12
        },
        'title_font': {
            'family': 'Rajdhani, sans-serif',
            'size': 16
        }
    }


def create_gravitational_lensing_figure(result):
    """Algorithm #11: Gravitational lensing map"""
    try:
        curvature = np.array(result['gravitational_map'])
        
        fig = go.Figure(data=go.Heatmap(
            z=curvature,
            colorscale='Viridis',
            colorbar=dict(title='Curvature')
        ))
        
        theme_config = get_plotly_theme_config()
        fig.update_layout(
            title='Gravitational Lensing Map (Spacetime Curvature)',
            xaxis_title='Position X',
            yaxis_title='Position Y',
            height=400,
            **theme_config
        )
        
        return fig
    except:
        return go.Figure()


def create_epigenetic_heatmap(result):
    """Algorithm #12: Methylation heatmap"""
    try:
        cpg_density = result['cpg_density']
        accessibility = result['accessibility_profile']
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('CpG Island Density (Methylation Sites)', 
                          'Chromatin Accessibility (Open vs Closed)')
        )
        
        fig.add_trace(
            go.Scatter(y=cpg_density, mode='lines', name='CpG Density',
                      line=dict(color='#00d4ff', width=2)),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(y=accessibility, mode='lines', name='Accessibility',
                      line=dict(color='#00ff88', width=2)),
            row=2, col=1
        )
        
        theme_config = get_plotly_theme_config()
        fig.update_layout(height=500, showlegend=True, **theme_config)
        return fig
    except:
        return go.Figure()


def create_quantum_interference_figure(result):
    """Algorithm #13: Quantum interference pattern"""
    try:
        probabilities = result['probability_distribution']
        
        # Create 3D bar chart effect
        fig = go.Figure()
        for i, prob in enumerate(probabilities):
            fig.add_trace(go.Scatter3d(
                x=[i, i],
                y=[0, prob],
                z=[i, i],
                mode='lines+markers',
                line=dict(color=px.colors.sample_colorscale('Plasma', prob)[0], width=12),
                marker=dict(size=6, color=px.colors.sample_colorscale('Plasma', prob)[0]),
                showlegend=False
            ))
        
        theme_config = get_plotly_theme_config()
        fig.update_layout(
            title='Quantum Interference Pattern (3D)',
            scene=dict(
                xaxis_title='Node State',
                yaxis_title='Quantum Probability',
                zaxis_title='Index'
            ),
            height=500,
            **theme_config
        )
        
        return fig
    except:
        return go.Figure()


def create_fluid_dynamics_figure(result):
    """Algorithm #14: Fluid flow visualization"""
    try:
        vx = np.array(result['velocity_field_x'])
        vy = np.array(result['velocity_field_y'])
        vorticity = np.array(result['vorticity_map'])
        
        # Create quiver plot for velocity field
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Velocity Field (Data Flow)', 'Vorticity (Turbulence)')
        )
        
        # Downsample for visualization
        step = max(1, len(vx) // 20)
        Y, X = np.mgrid[0:vx.shape[0]:step, 0:vx.shape[1]:step]
        
        fig.add_trace(
            go.Heatmap(z=vorticity, colorscale='RdBu', showscale=True),
            row=1, col=2
        )
        
        # Velocity magnitude
        v_mag = np.sqrt(vx**2 + vy**2)
        fig.add_trace(
            go.Heatmap(z=v_mag, colorscale='Jet', showscale=True),
            row=1, col=1
        )
        
        theme_config = get_plotly_theme_config()
        fig.update_layout(height=400, **theme_config)
        return fig
    except:
        return go.Figure()


def create_stylometric_radar_figure(result):
    """Algorithm #15: Radar chart for stylometric dimensions"""
    try:
        dimensions = result['stylometric_dimensions']
        
        categories = list(dimensions.keys())
        values = list(dimensions.values())
        
        fig = go.Figure(data=go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            line=dict(color='#ff3366', width=2)
        ))
        
        theme_config = get_plotly_theme_config()
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1])
            ),
            title='Stylometric Phonetic Radar (Code Fingerprint)',
            height=500,
            **theme_config
        )
        
        return fig
    except:
        return go.Figure()


def create_entropy_surface_figure(result):
    """Algorithm #16: 3D entropy surface"""
    try:
        entropy_surface = np.array(result['entropy_surface'])
        
        fig = go.Figure(data=go.Surface(
            z=entropy_surface,
            colorscale='Hot',
            colorbar=dict(title='Entropy')
        ))
        
        theme_config = get_plotly_theme_config()
        fig.update_layout(
            title='Event Horizon Entropy Surface (Information Barriers)',
            scene=dict(
                xaxis_title='Position X',
                yaxis_title='Position Y',
                zaxis_title='Entropy'
            ),
            height=500,
            **theme_config
        )
        
        return fig
    except:
        return go.Figure()


def create_symbiotic_graph_figure(result):
    """Algorithm #17: Ecological relationship graph"""
    try:
        relationships = result['relationships']
        
        # Create 3D bar chart of relationship types
        categories = list(relationships.keys())
        counts = list(relationships.values())
        
        colors = ['#00ff88', '#ff3366', '#ff9933', '#ffdd00', '#00d4ff']
        
        fig = go.Figure()
        for i, (cat, count, color) in enumerate(zip(categories, counts, colors)):
            fig.add_trace(go.Scatter3d(
                x=[i, i],
                y=[0, count],
                z=[i*0.5, i*0.5],
                mode='lines+markers',
                line=dict(color=color, width=15),
                marker=dict(size=8, color=color),
                name=cat,
                showlegend=True
            ))
        
        theme_config = get_plotly_theme_config()
        fig.update_layout(
            title='Symbiotic Process Relationships (3D)',
            scene=dict(
                xaxis_title='Category Index',
                yaxis_title='Count',
                zaxis_title='Depth'
            ),
            height=500,
            **theme_config
        )
        
        return fig
    except:
        return go.Figure()


def create_temporal_manifold_figure(result):
    """Algorithm #18: 4D hypercube projection"""
    try:
        projection = np.array(result['hypercube_projection'])
        
        if len(projection) > 0:
            fig = go.Figure(data=go.Scatter3d(
                x=projection[:, 0],
                y=projection[:, 1],
                z=projection[:, 2],
                mode='markers',
                marker=dict(
                    size=3,
                    color=np.arange(len(projection)),
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title='Time')
                )
            ))
            
            theme_config = get_plotly_theme_config()
            fig.update_layout(
                title='Chrono-Slicing Temporal Manifold (4D → 3D Projection)',
                scene=dict(
                    xaxis_title='Dimension 1',
                    yaxis_title='Dimension 2',
                    zaxis_title='Dimension 3'
                ),
                height=500,
                **theme_config
            )
            
            return fig
    except:
        pass
    
    return go.Figure()


def create_verification_gauge_figure(result):
    """Algorithm #19: Verification confidence gauge"""
    try:
        hybrid_conf = result['hybrid_confidence']
        neural_conf = result['neural_confidence']
        symbolic_conf = result['symbolic_confidence']
        
        fig = make_subplots(
            rows=1, cols=3,
            specs=[[{'type': 'indicator'}, {'type': 'indicator'}, {'type': 'indicator'}]],
            subplot_titles=('Hybrid', 'Neural', 'Symbolic')
        )
        
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=hybrid_conf * 100,
            title={'text': "Hybrid"},
            gauge={'axis': {'range': [0, 100]},
                   'bar': {'color': "#00d4ff"},
                   'steps': [
                       {'range': [0, 30], 'color': "#00ff88"},
                       {'range': [30, 60], 'color': "#ffdd00"},
                       {'range': [60, 100], 'color': "#ff3366"}
                   ]}
        ), row=1, col=1)
        
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=neural_conf * 100,
            title={'text': "Neural"},
            gauge={'axis': {'range': [0, 100]},
                   'bar': {'color': "#9966ff"}}
        ), row=1, col=2)
        
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=symbolic_conf * 100,
            title={'text': "Symbolic"},
            gauge={'axis': {'range': [0, 100]},
                   'bar': {'color': "#00ff88"}}
        ), row=1, col=3)
        
        theme_config = get_plotly_theme_config()
        fig.update_layout(
            title='Neural-Symbolic Verification Confidence',
            height=300,
            **theme_config
        )
        
        return fig
    except:
        return go.Figure()


def create_audio_waveform_figure(result):
    """Algorithm #20: Audio waveform + spectrogram"""
    try:
        audio_signal = result['audio_signal']
        time_points = result['time_points']
        
        # Spectrogram data
        spec_freq = result['spectrogram_freq']
        spec_time = result['spectrogram_time']
        spec_power = np.array(result['spectrogram_power'])
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Audio Waveform (Sonified Code)', 
                          'Spectrogram (Time-Frequency Analysis)')
        )
        
        # Waveform
        fig.add_trace(
            go.Scatter(x=time_points, y=audio_signal, mode='lines',
                      line=dict(color='#00d4ff', width=1)),
            row=1, col=1
        )
        
        # Spectrogram
        fig.add_trace(
            go.Heatmap(x=spec_time, y=spec_freq, z=spec_power,
                      colorscale='Hot', showscale=True),
            row=2, col=1
        )
        
        fig.update_xaxes(title_text="Time (s)", row=1, col=1)
        fig.update_xaxes(title_text="Time (s)", row=2, col=1)
        fig.update_yaxes(title_text="Amplitude", row=1, col=1)
        fig.update_yaxes(title_text="Frequency (Hz)", row=2, col=1)
        
        theme_config = get_plotly_theme_config()
        fig.update_layout(height=600, showlegend=False, **theme_config)
        return fig
    except:
        return go.Figure()


# Mapping of algorithm IDs to visualization functions
VISUALIZATION_MAP = {
    11: create_gravitational_lensing_figure,
    12: create_epigenetic_heatmap,
    13: create_quantum_interference_figure,
    14: create_fluid_dynamics_figure,
    15: create_stylometric_radar_figure,
    16: create_entropy_surface_figure,
    17: create_symbiotic_graph_figure,
    18: create_temporal_manifold_figure,
    19: create_verification_gauge_figure,
    20: create_audio_waveform_figure
}


def get_visualization(algo_id, result):
    """Get visualization for a specific algorithm"""
    if algo_id in VISUALIZATION_MAP:
        try:
            return VISUALIZATION_MAP[algo_id](result)
        except Exception as e:
            print(f"Visualization error for algorithm {algo_id}: {e}")
            return go.Figure()
    return go.Figure()

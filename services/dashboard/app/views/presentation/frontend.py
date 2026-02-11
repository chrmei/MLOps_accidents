"""Frontend & User Interface - Presentation Page (Christian - Detailed)."""

import streamlit as st


def render():
    """Render Frontend page with detailed graphics and overviews."""
    st.title("Frontend & User Interface")
    st.markdown("**Duration: 2-3 minutes** | **Presenter: Christian**")
    st.markdown("---")

    # Streamlit Dashboard Overview
    st.header("Streamlit Dashboard")
    st.markdown("**Port: 8501**")
    
    # Dashboard Architecture
    st.subheader("Dashboard Architecture")
    
    arch_col1, arch_col2 = st.columns([1, 1])
    
    with arch_col1:
        st.markdown("""
        <div style="padding: 20px; background-color: #2d2d3d; border-radius: 10px;">
            <h4 style="color: #e63946;">🔐 Authentication Layer</h4>
            <p>Integrated with Auth Service API</p>
            <ul>
                <li>JWT token management</li>
                <li>Session handling</li>
                <li>Role-based access</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with arch_col2:
        st.markdown("""
        <div style="padding: 20px; background-color: #2d2d3d; border-radius: 10px;">
            <h4 style="color: #e63946;">👥 User Roles</h4>
            <p><strong>Regular Users:</strong></p>
            <ul>
                <li>Prediction form</li>
                <li>Real-time predictions</li>
            </ul>
            <p><strong>Admin Users:</strong></p>
            <ul>
                <li>Data Ops</li>
                <li>ML Ops</li>
                <li>User Management</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # User Features
    st.header("User Features")
    
    user_col1, user_col2 = st.columns(2)
    
    with user_col1:
        st.markdown("""
        <div style="padding: 15px; background-color: #1e1e2e; border-radius: 5px; border-left: 4px solid #00ff00;">
            <h4>🎯 Interactive Prediction Form</h4>
            <ul>
                <li>Address input with geocoding</li>
                <li>Accident details form</li>
                <li>Real-time validation</li>
                <li>Weather data integration</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with user_col2:
        st.markdown("""
        <div style="padding: 15px; background-color: #1e1e2e; border-radius: 5px; border-left: 4px solid #00ff00;">
            <h4>⚡ Real-Time Prediction</h4>
            <ul>
                <li>Instant severity prediction</li>
                <li>Probability scores</li>
                <li>Confidence indicators</li>
                <li>Result visualization</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Admin Features
    st.header("Admin Features")
    
    admin_col1, admin_col2, admin_col3 = st.columns(3)
    
    with admin_col1:
        st.markdown("""
        <div style="padding: 15px; background-color: #1e1e2e; border-radius: 5px; border-left: 4px solid #ffa500;">
            <h4>📊 Data Ops</h4>
            <ul>
                <li>Trigger preprocessing</li>
                <li>Run feature engineering</li>
                <li>View job logs</li>
                <li>Monitor job status</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with admin_col2:
        st.markdown("""
        <div style="padding: 15px; background-color: #1e1e2e; border-radius: 5px; border-left: 4px solid #ffa500;">
            <h4>🤖 ML Ops</h4>
            <ul>
                <li>Trigger training jobs</li>
                <li>View training metrics</li>
                <li>Manage model config</li>
                <li>Monitor training progress</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with admin_col3:
        st.markdown("""
        <div style="padding: 15px; background-color: #1e1e2e; border-radius: 5px; border-left: 4px solid #ffa500;">
            <h4>👤 User Management</h4>
            <ul>
                <li>Create users</li>
                <li>Update user roles</li>
                <li>Delete users</li>
                <li>View user list</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Authentication Flow
    st.header("Authentication Flow")
    
    auth_flow = """
    ```
    User Login
        ↓
    Auth Service (port 8004)
        ↓
    JWT Token Generated
        ↓
    Token Stored in Session
        ↓
    API Requests Include Token
        ↓
    Role-Based Access Control
        ↓
    Dashboard Features Unlocked
    ```
    """
    st.code(auth_flow, language="text")

    st.markdown("---")

    # Role-Based Access
    st.header("Role-Based Access Control")
    
    role_col1, role_col2 = st.columns(2)
    
    with role_col1:
        st.markdown("""
        <div style="padding: 15px; background-color: #2d2d3d; border-radius: 5px;">
            <h4>👤 Regular User</h4>
            <p><strong>Access:</strong></p>
            <ul>
                <li>✅ Prediction form</li>
                <li>✅ View predictions</li>
                <li>❌ Data operations</li>
                <li>❌ ML operations</li>
                <li>❌ User management</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with role_col2:
        st.markdown("""
        <div style="padding: 15px; background-color: #2d2d3d; border-radius: 5px;">
            <h4>🔧 Admin User</h4>
            <p><strong>Access:</strong></p>
            <ul>
                <li>✅ Prediction form</li>
                <li>✅ View predictions</li>
                <li>✅ Data operations</li>
                <li>✅ ML operations</li>
                <li>✅ User management</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

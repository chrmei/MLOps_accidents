"""Introduction & Problem Statement - Presentation Page (Rafael)."""

import streamlit as st


def render():
    """Render Introduction & Problem Statement page."""
    st.title("Introduction & Overview")
    st.markdown("**Duration: 2-3 minutes** | **Presenter: Rafael**")
    st.markdown("---")



    st.header("Problem Statement")
    st.markdown(
        """
        <div style="text-align: center; padding: 15px;">
            <span style="font-size: 1.3rem; font-weight: 700;">
                Predicting road accident severity using French road accident data
            </span>
        </div>
        """, unsafe_allow_html=True)

    
    st.subheader("**Data**")
    st.markdown(
        """
        - Provided by the [French government](https://www.data.gouv.fr/datasets/bases-de-donnees-annuelles-des-accidents-corporels-de-la-circulation-routiere-annees-de-2005-a-2024),
        also available in [Kaggle](https://www.kaggle.com/datasets/ahmedlahlou/accidents-in-france-from-2005-to-2016)
        - Consists of 4 raw CSV files
        """
    )
    st.markdown(
        """
        <div style="display: flex; gap: 20px;">
            <div style="padding: 15px; background-color: #336086; border-radius: 8px; width: 45%;">
                <h3 style="text-align:center;">Victim Data</h3>
                <ul>
                    <li>safety equipment</li>
                    <li>position in vehicle</li>
                    <li>etc.</li>
                </ul>
            </div>
            <div style="padding: 15px; background-color: #336086; border-radius: 8px; width: 45%;">
                <h3 style="text-align:center;">Accident Data</h3>
                <ul>
                    <li>time</li>
                    <li>coordinates</li>
                    <li>etc.</li>
                </ul>
            </div>
            <div style="padding: 15px; background-color: #336086; border-radius: 8px; width: 45%;">
                <h3 style="text-align:center;">Location Data</h3>
                <ul>
                    <li>road type</li>
                    <li>surface</li>
                    <li>etc.</li>
                </ul>
            </div>
            <div style="padding: 15px; background-color: #336086; border-radius: 8px; width: 45%;">
                <h3 style="text-align:center;">Vehicule Data</h3>
                <ul>
                    <li>type</li>
                    <li>maneuver</li>
                    <li>etc.</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown(" ")
    st.subheader("**Target Label**")
    st.markdown(
        """
        - Derived from severity levels given for each victim
        """
    )
    st.markdown(
        """
        <div style="display: flex; gap: 20px; text-align: center;">
            <div style="padding: 15px; background-color: #336086; border-radius: 8px; width: 45%;">
                <h3>0</h3>
                <p>Victims only slightly injured</p>
            </div>
            <div style="padding: 15px; background-color: #336086; border-radius: 8px; width: 45%;">
                <h3>1</h3>
                <p>At least one victim seriously injured or dead</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.header("Objective")
    st.markdown(
        """
        <div style="text-align: center; padding: 15px;">
            <span style="font-size: 1.3rem; font-weight: 700;">
                Build a <b>reproducible</b>, <b>scalable</b>, and <b>maintainable</b> ML product for accident severity prediction
            </span>
        </div>
        """, unsafe_allow_html=True)

    st.subheader("Key Components")
    st.markdown(
        """
        📐 **System Architecture** - Surya

        🔁 **Data Science & ML Pipeline** - Christian

        🖥️ **Frontend & User Interface** - Christian

        📊 **Monitoring & Observability** - Rafael

        ⚙️ **DevOps & Automation** - Surya

        🚀 **Conclusion & Future Work** - Rafael
        """
    )

import streamlit as st

st.set_page_config(
    page_title="Team - Cloud Resource Optimizer",
    page_icon="👥",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .team-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(120deg, #1a365d 0%, #2a4365 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .team-card {
        background: linear-gradient(145deg, #2d3748 0%, #1a202c 100%);
        color: #ffffff;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        margin-bottom: 1.5rem;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        text-align: center;
        border: 1px solid #4a5568;
    }
    .team-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.3);
        border-color: #63b3ed;
    }
    .team-card h3 {
        color: #63b3ed;
        margin-bottom: 0.5rem;
        font-size: 1.5rem;
    }
    .role-badge {
        background-color: #2b6cb0;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.9rem;
        margin: 0.5rem 0;
        display: inline-block;
        border: 1px solid #4299e1;
    }
    .department {
        color: #a0aec0;
        font-style: italic;
        margin: 0.5rem 0;
    }
    .social-links {
        margin-top: 1.5rem;
        padding-top: 1rem;
        border-top: 1px solid #4a5568;
    }
    .social-links a {
        color: #63b3ed;
        margin: 0 0.5rem;
        text-decoration: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        transition: all 0.3s ease;
    }
    .social-links a:hover {
        color: #90cdf4;
        background-color: #2d3748;
    }
    .team-card p {
        color: #e2e8f0;
        line-height: 1.6;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
    <div class="team-header">
        <h1>Meet Our Team</h1>
        <p>The brilliant minds behind the Cloud Resource Optimizer</p>
    </div>
""", unsafe_allow_html=True)

# Team Introduction
st.markdown("""
## 🌟 Our Vision
Our team combines expertise in cloud computing, machine learning, and software engineering to create
an innovative solution for cloud resource optimization. Each member brings unique skills and perspectives
to deliver a cutting-edge product.
""")

# Team Members
st.header("👥 Core Team Members")

# Create three columns for team members
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
        <div class="team-card">
            <h3>Shravi Magdum</h3>
            <p class="department">CSE (Data Science)</p>
            <div class="role-badge">ML Model Development</div>
            <div class="social-links">
                <a href="#"><i class="fab fa-linkedin"></i> LinkedIn</a>
                <a href="#"><i class="fab fa-github"></i> GitHub</a>
            </div>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
        <div class="team-card">
            <h3>Palak Mundada</h3>
            <p class="department">CSE (Data Science)</p>
            <div class="role-badge">ML Model Development</div>
            <div class="social-links">
                <a href="#"><i class="fab fa-linkedin"></i> LinkedIn</a>
                <a href="#"><i class="fab fa-github"></i> GitHub</a>
            </div>
        </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
        <div class="team-card">
            <h3>Radha Kulkarni</h3>
            <p class="department">CSE (Data Science)</p>
            <div class="role-badge">Backend Development</div>
            <div class="social-links">
                <a href="#"><i class="fab fa-linkedin"></i> LinkedIn</a>
                <a href="#"><i class="fab fa-github"></i> GitHub</a>
            </div>
        </div>
    """, unsafe_allow_html=True)

# Second row of team members
col4, col5, col6 = st.columns(3)

with col4:
    st.markdown("""
        <div class="team-card">
            <h3>Parth Shinge</h3>
            <p class="department">CSE (Data Science)</p>
            <div class="role-badge">UI/UX Development</div>
            <div class="social-links">
                <a href="#"><i class="fab fa-linkedin"></i> LinkedIn</a>
                <a href="#"><i class="fab fa-github"></i> GitHub</a>
            </div>
        </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown("""
        <div class="team-card">
            <h3>Nrusinha Mane</h3>
            <p class="department">CSE (Data Science)</p>
            <div class="role-badge">Research & Development</div>
            <div class="social-links">
                <a href="#"><i class="fab fa-linkedin"></i> LinkedIn</a>
                <a href="#"><i class="fab fa-github"></i> GitHub</a>
            </div>
        </div>
    """, unsafe_allow_html=True)

with col6:
    st.markdown("""
        <div class="team-card">
            <h3>Vivek Shirsath</h3>
            <p class="department">CSE (Data Science)</p>
            <div class="role-badge">Research & Development</div>
            <div class="social-links">
                <a href="#"><i class="fab fa-linkedin"></i> LinkedIn</a>
                <a href="#"><i class="fab fa-github"></i> GitHub</a>
            </div>
        </div>
    """, unsafe_allow_html=True)



# Contact Section
st.header("📬 Get in Touch")
st.markdown("""
If you'd like to learn more about our project or collaborate with us, please reach out:



- 📍 Location: Vishwakarma Institute of Technology
""")

# Footer

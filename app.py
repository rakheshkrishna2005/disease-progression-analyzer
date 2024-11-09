import streamlit as st
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, AIMessage
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize Streamlit page configuration
st.set_page_config(page_title="Disease Progression Analyzer", page_icon="🏥")
st.markdown("<h1 style='text-align: center;'>🏥 AI Medical Assistant</h1>", unsafe_allow_html=True)

# Define custom instruction template
CUSTOM_PROMPT_TEMPLATE = """
### Role and Context
You are a specialized medical AI assistant focused on analyzing neurodegenerative disease progression. Your task is to analyze the provided patient data and generate a comprehensive clinical assessment report.

### Response Structure
Generate your response in the following format:

1. CLINICAL ASSESSMENT SUMMARY
- Current Disease Stage
- Key Risk Factors
- Primary Concerns

2. DETAILED ANALYSIS
A. Cognitive Status
- MMSE Score Interpretation
- ADAS-Cog Score Analysis
- CDR Score Assessment
- Overall Cognitive Trajectory

B. Neuroimaging Findings
- Hippocampal Volume Analysis
- Cortical Thickness Evaluation
- PET Scan Interpretations
- Brain Atrophy Patterns

C. Biomarker Analysis
- CSF Biomarker Interpretation
- Blood Biomarker Assessment
- Correlation with Clinical Findings

D. Risk Factor Assessment
- Genetic Risk Analysis
- Family History Impact
- Modifiable Risk Factors
- Lifestyle Factor Evaluation

3. DISEASE PROGRESSION INDICATORS
- Current Progression Rate
- Key Progression Markers
- Warning Signs to Monitor

4. RECOMMENDATIONS
- Suggested Monitoring Schedule
- Key Areas for Intervention
- Lifestyle Modifications
- Additional Tests/Assessments (if needed)

### Instructions
- Provide specific, quantitative interpretations where possible
- Compare values to known clinical benchmarks
- Highlight any concerning deviations from normal ranges
- Use medical terminology while maintaining clarity
- Include evidence-based predictions about disease trajectory
- Flag any urgent concerns requiring immediate attention

### Current conversation:
{chat_history}

### Patient Data:
{human_input}

### Assistant Response:
"""

prompt = PromptTemplate(
    input_variables=["chat_history", "human_input"],
    template=CUSTOM_PROMPT_TEMPLATE
)

# Initialize the Groq LLM
def initialize_llm():
    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama3-8b-8192",
        temperature=0.7,
        max_tokens=4096,
    )
    return llm

# Instantiate the LLM globally
llm = initialize_llm()

# Initialize session states
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        return_messages=True,
        memory_key="chat_history"
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

# Demographic and Medical Information Form
with st.form("patient_info"):
    st.markdown("<h3 style='text-align: center;'>Neurodegenerative Disease Assessment</h3>", unsafe_allow_html=True)
    
    # Basic Information
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=0, max_value=120, value=65, step=1)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"], index=0)
        education = st.selectbox("Education Level", ["< 12 years", "12-16 years", "> 16 years"], index=1)
        occupation = st.text_input("Occupation (Current/Former)", value="Retired Teacher")
    
    with col2:
        disease_duration = st.number_input("Disease Duration (years)", min_value=0, max_value=30, value=2, step=1)
        age_of_onset = st.number_input("Age of First Symptoms", min_value=0, max_value=120, value=63, step=1)
        handedness = st.selectbox("Handedness", ["Right", "Left", "Ambidextrous"], index=0)
    
    # Cognitive and Behavioral Assessment
    st.markdown("<h3 style='text-align: center;'>Cognitive and Behavioral Assessment</h3>", unsafe_allow_html=True)
    col3, col4 = st.columns(2)
    
    with col3:
        mmse_score = st.number_input("MMSE Score (0-30)", min_value=0, max_value=30, value=26, step=1)
        moca_score = st.number_input("MoCA Score (0-30)", min_value=0, max_value=30, value=24, step=1)
        cdr_score = st.selectbox("Clinical Dementia Rating (CDR)", 
                               ["0 (Normal)", "0.5 (Very Mild)", "1 (Mild)", "2 (Moderate)", "3 (Severe)"], 
                               index=1)
    
    with col4:
        faq_score = st.number_input("Functional Activities Questionnaire (FAQ) Score (0-30)", 
                                   min_value=0, max_value=30, value=5, step=1)
        gds_score = st.number_input("Geriatric Depression Scale (0-15)", 
                                   min_value=0, max_value=15, value=3, step=1)
        npi_score = st.number_input("Neuropsychiatric Inventory Score (0-144)", 
                                   min_value=0, max_value=144, value=10, step=1)
    
    # Motor Symptoms
    st.markdown("<h3 style='text-align: center;'>Motor Symptoms</h3>", unsafe_allow_html=True)
    col5, col6 = st.columns(2)
    
    with col5:
        gait_speed = st.number_input("Gait Speed (meters/second)", 
                                    min_value=0.0, max_value=2.0, value=1.0, step=0.1)
        balance_score = st.selectbox("Balance Assessment", 
                                   ["Normal", "Mild Impairment", "Moderate Impairment", "Severe Impairment"],
                                   index=1)
        falls_6months = st.number_input("Number of Falls (past 6 months)", 
                                      min_value=0, max_value=50, value=1, step=1)
    
    with col6:
        tremor = st.selectbox("Tremor Presence", ["None", "Mild", "Moderate", "Severe"], index=0)
        rigidity = st.selectbox("Rigidity", ["None", "Mild", "Moderate", "Severe"], index=0)
        bradykinesia = st.selectbox("Bradykinesia", ["None", "Mild", "Moderate", "Severe"], index=1)
    
    # Biomarkers
    st.markdown("<h3 style='text-align: center;'>Biomarkers and Imaging</h3>", unsafe_allow_html=True)
    col7, col8 = st.columns(2)
    
    with col7:
        # CSF Biomarkers
        csf_abeta = st.number_input("CSF Aβ42 (pg/mL)", 
                                   min_value=0, max_value=2000, value=500, step=10)
        csf_tau = st.number_input("CSF Total Tau (pg/mL)", 
                                 min_value=0, max_value=2000, value=350, step=10)
        csf_ptau = st.number_input("CSF Phosphorylated Tau (pg/mL)", 
                                  min_value=0, max_value=200, value=60, step=5)
        nfl_level = st.number_input("Serum Neurofilament Light Chain (pg/mL)", 
                                   min_value=0, max_value=200, value=22, step=1)
    
    with col8:
        # Imaging Markers
        hippocampal_volume = st.number_input("Hippocampal Volume (mm³)", 
                                           min_value=1000, max_value=5000, value=3500, step=100)
        ventricle_volume = st.number_input("Ventricular Volume (mm³)", 
                                         min_value=10000, max_value=100000, value=35000, step=1000)
        cortical_thickness = st.number_input("Mean Cortical Thickness (mm)", 
                                           min_value=1.0, max_value=5.0, value=2.5, step=0.1)
    
    # Genetic and Risk Factors
    st.markdown("<h3 style='text-align: center;'>Genetic and Risk Factors</h3>", unsafe_allow_html=True)
    col9, col10 = st.columns(2)
    
    with col9:
        apoe_status = st.selectbox("APOE Genotype", 
                                 ["ε3/ε3", "ε3/ε4", "ε4/ε4", "ε2/ε3", "ε2/ε4", "Unknown"], 
                                 index=1)
        family_history = st.multiselect("Family History", 
                                      ["Alzheimer's", "Parkinson's", "FTD", "ALS", "Other Neurological"],
                                      default=["Alzheimer's"])
        cardiovascular_risk = st.multiselect("Cardiovascular Risk Factors",
                                           ["Hypertension", "Diabetes", "Hyperlipidemia", "Smoking", "Obesity"],
                                           default=["Hypertension"])
    
    with col10:
        sleep_disorders = st.multiselect("Sleep Disorders",
                                       ["None", "Sleep Apnea", "REM Behavior Disorder", "Insomnia"],
                                       default=["None"])
        traumatic_brain_injury = st.selectbox("History of Traumatic Brain Injury",
                                            ["None", "Mild", "Moderate", "Severe"],
                                            index=0)
        psychiatric_history = st.multiselect("Psychiatric History",
                                           ["None", "Depression", "Anxiety", "Bipolar", "Schizophrenia"],
                                           default=["None"])
    
    # Lifestyle and Environmental Factors
    st.markdown("<h3 style='text-align: center;'>Lifestyle and Environmental Factors</h3>", unsafe_allow_html=True)
    col11, col12 = st.columns(2)
    
    with col11:
        physical_activity = st.select_slider("Physical Activity (hours/week)",
                                           options=[0, 1, 2, 3, 4, 5, 6, 7, '7+'],
                                           value=3)
        cognitive_engagement = st.select_slider("Cognitive Activities (hours/week)",
                                             options=[0, 1, 2, 3, 4, 5, 6, 7, '7+'],
                                             value=4)
        social_engagement = st.select_slider("Social Activities (hours/week)",
                                          options=[0, 1, 2, 3, 4, 5, 6, 7, '7+'],
                                          value=3)
    
    with col12:
        diet_pattern = st.multiselect("Diet Pattern",
                                    ["Mediterranean", "MIND Diet", "Western", "Vegetarian", "Other"],
                                    default=["Mediterranean"])
        sleep_quality = st.select_slider("Sleep Quality",
                                       options=["Poor", "Fair", "Good", "Excellent"],
                                       value="Good")
        stress_level = st.select_slider("Stress Level",
                                      options=["Low", "Moderate", "High", "Severe"],
                                      value="Moderate")

    # Submit button
    submitted = st.form_submit_button("Analyze Disease Progression")
    
    if submitted:
        # Aggregate all details into a structured message
        patient_details = f"""
        BASIC INFORMATION:
        Age: {age} | Gender: {gender} | Education: {education}
        Disease Duration: {disease_duration} years | Age of Onset: {age_of_onset}
        Occupation: {occupation} | Handedness: {handedness}
        
        COGNITIVE AND BEHAVIORAL SCORES:
        MMSE: {mmse_score}/30 | MoCA: {moca_score}/30 | CDR: {cdr_score}
        FAQ: {faq_score}/30 | GDS: {gds_score}/15 | NPI: {npi_score}/144
        
        MOTOR SYMPTOMS:
        Gait Speed: {gait_speed} m/s | Balance: {balance_score}
        Falls (6 months): {falls_6months}
        Tremor: {tremor} | Rigidity: {rigidity} | Bradykinesia: {bradykinesia}
        
        BIOMARKERS:
        CSF Aβ42: {csf_abeta} pg/mL | Total Tau: {csf_tau} pg/mL
        Phosphorylated Tau: {csf_ptau} pg/mL | NFL: {nfl_level} pg/mL
        
        IMAGING MARKERS:
        Hippocampal Volume: {hippocampal_volume} mm³
        Ventricular Volume: {ventricle_volume} mm³
        Cortical Thickness: {cortical_thickness} mm
        
        GENETIC AND RISK FACTORS:
        APOE: {apoe_status}
        Family History: {', '.join(family_history)}
        Cardiovascular Risks: {', '.join(cardiovascular_risk)}
        Sleep Disorders: {', '.join(sleep_disorders)}
        TBI History: {traumatic_brain_injury}
        Psychiatric History: {', '.join(psychiatric_history)}
        
        LIFESTYLE FACTORS:
        Physical Activity: {physical_activity} hrs/week
        Cognitive Engagement: {cognitive_engagement} hrs/week
        Social Engagement: {social_engagement} hrs/week
        Diet: {', '.join(diet_pattern)}
        Sleep Quality: {sleep_quality}
        Stress Level: {stress_level}
        """
        
        # Add user message to chat history
        st.session_state.messages.append(HumanMessage(content=patient_details))
        
        # Format chat history
        chat_history = "\n".join([f"{'Human' if isinstance(msg, HumanMessage) else 'Assistant'}: {msg.content}" 
                                  for msg in st.session_state.messages[:-1]])
        
        # Create the formatted prompt
        formatted_prompt = prompt.format(
            chat_history=chat_history,
            human_input=patient_details
        )
        
        # Get AI response
        with st.spinner("Analyzing disease progression..."):
            response = llm.invoke(formatted_prompt)
            st.session_state.messages.append(AIMessage(content=response.content))
            st.session_state.analysis_result = response.content
            st.write(response.content)

# Initialize chat message history
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

# Chat Interface
st.markdown("<h2 style='text-align: center;'>💬 Chat with AI Medical Assistant</h2>", unsafe_allow_html=True)

# Display chat messages using Streamlit's chat interface
for message in st.session_state.chat_messages:
    if message["role"] == "user":
        st.chat_message("user").write(message["content"])
    else:
        st.chat_message("assistant").write(message["content"])

# Chat input using Streamlit's chat input
if prompt := st.chat_input("Ask more about the analysis..."):
    # Display user message
    st.chat_message("user").write(prompt)
    
    # Add message to chat history
    st.session_state.chat_messages.append({"role": "user", "content": prompt})
    
    # Format the entire conversation history
    chat_history = "\n".join([f"{'Human' if isinstance(msg, HumanMessage) else 'Assistant'}: {msg.content}" 
                             for msg in st.session_state.messages])
    
    # Create a new prompt that includes the analysis context
    chat_prompt = f"""
    Based on the previous analysis and conversation, please answer the following question:
    
    Previous analysis:
    {st.session_state.analysis_result}
    
    Question: {prompt}
    """
    
    # Get AI response
    with st.spinner("Thinking..."):
        response = llm.invoke(chat_prompt)
        
        # Display assistant response
        st.chat_message("assistant").write(response.content)
        
        # Add response to chat history
        st.session_state.chat_messages.append({"role": "assistant", "content": response.content})

# Clear conversation button
if st.button("Clear Chat"):
    st.session_state.chat_messages = []
    st.rerun()
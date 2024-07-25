import json
import random
from gpt_api import ChatGPT

medical_elements = [
    "Blood Pressure", "Blood Glucose", "Heart Rate", "Cholesterol Level", "White Blood Cell Count",
    "Red Blood Cell Count", "Platelet Count", "Creatinine", "Blood Urea Nitrogen", "Liver Function Tests",
    "Kidney Function Tests", "Electrolytes", "Blood Lipid Profile", "Urinalysis", "Complete Blood Count",
    "CT Scan", "MRI Scan", "X-ray", "Ultrasound", "ECG", "EEG", "Endoscopy",
    "Colonoscopy", "Aspirin", "Penicillin", "Cephalosporins", "Erythromycin", "Gentamicin", "Insulin", "Amoxicillin",
    "Antipyrine", "Ibuprofen", "Acetaminophen", "Codeine", "Morphine", "Hydrocodone", "Leukemia", "Lymphoma",
    "Hypertension", "Diabetes Mellitus", "Cardiovascular Disease", "Chronic Obstructive Pulmonary Disease", "Alzheimer's Disease",
    "Parkinson's Disease", "Cancer", "Influenza", "Pneumonia", "Tuberculosis", "HIV/AIDS", "Malaria", "Cholera",
    "Measles", "Rubella", "Varicella", "Hepatitis B", "Hepatitis C", "Menigitis", "Encephalitis", "Septicemia",
    "Anemia", "Thrombocytopenia", "Coagulation Disorders", "Hemophilia", "Dyslipidemia", "Obesity", "Osteoporosis",
    "Arthritis", "Rheumatoid Arthritis", "Osteoarthritis", "Multiple Sclerosis", "Spinal Cord Injury", "Stroke", "Traumatic Brain Injury",
    "Cataract", "Glaucoma", "Macular Degeneration", "Diabetic Retinopathy", "Hyperopia", "Myopia", "Astigmatism",
    "Hearing Loss", "Tinnitus", "Vertigo", "Allergy", "Asthma", "Bronchitis", "Emphysema", "Pneumonitis",
    "Gastroesophageal Reflux Disease", "Peptic Ulcer Disease", "Irritable Bowel Syndrome", "Inflammatory Bowel Disease",
    "Cirrhosis", "Pancreatitis", "Hepatitis", "Cholecystitis", "Gallstones", "Kidney Stones", "Urinary Tract Infection",
    "Pyelonephritis", "Endocarditis", "Cardiomyopathy", "Arrhythmia", "Congestive Heart Failure", "Valvular Heart Disease",
    "Aortic Aneurysm", "Peripheral Artery Disease", "Deep Vein Thrombosis", "Pulmonary Embolism",
    "Carcinoma", "Sarcoma", "Melanoma", "Benign Tumors", "Metastasis", "Oncology", "Hematology", "Immunology",
    "Neurology", "Dermatology", "Ophthalmology", "Otorhinolaryngology", "Gastroenterology", "Pulmonology", "Cardiology",
    "Endocrinology", "Gynecology", "Obstetrics", "Pediatrics", "Urology", "Nephrology", "Hepatology", "Psychiatry",
    "Radiology", "Pathology", "Laboratory Medicine", "Pharmacology", "Toxicology", "Epidemiology", "Public Health",
    "Physiotherapy", "Occupational Therapy", "Speech Therapy", "Nutrition", "Dietetics", "Healthcare Management",
    "Health Informatics", "Medical Imaging", "Surgical Procedures", "Intensive Care Unit", "Emergency Medicine",
    "Family Medicine", "General Surgery", "Orthopedic Surgery", "Neurosurgery", "Cardiovascular Surgery", "Oncology Surgery",
    "Transplant Surgery", "Trauma Surgery", "Plastic Surgery", "Dental Surgery", "Ophthalmic Surgery", "ENT Surgery",
    "Gynecologic Surgery", "Pediatric Surgery", "Urologic Surgery", "Neurosurgical Procedures", "Cardiac Catheterization",
    "Joint Replacement", "Hip Fracture Repair", "Knee Surgery", "Laparoscopic Surgery", "Robotic Surgery", "Laser Surgery"
]
industrial_elements = [
    "Production Rate", "Energy Consumption", "Raw Material Cost", "Labor Cost", "Equipment Efficiency", "Product Quality", "Scrap Rate",
    "Automation Level", "Production Line Speed", "Inventory Turnover Rate", "Supply Chain Efficiency", "On-Time Delivery Rate", "Customer Satisfaction",
    "Carbon Emissions", "Wastewater Treatment Volume", "Exhaust Purification Rate", "Solid Waste Recycling Rate", "Environmental Impact Assessment", "Sustainability Indicators",
    "Steel", "Aluminum", "Plastics", "Rubber", "Glass", "Ceramics", "Composite Materials", "Semiconductor Materials",
    "Motor", "Pump", "Valve", "Sensor", "Actuator", "Controller", "Transformer", "Battery",
    "Industrial Robot", "CNC Machine", "3D Printer", "Laser Cutter", "Welding Robot", "Painting Equipment", "Testing Instrument",
    "Production Data", "Equipment Data", "Quality Data", "Energy Data", "Safety Data", "Environmental Data", "Economic Data",
    "ERP System", "MES System", "PLM System", "SCM System", "WMS System", "QMS System", "BMS System",
    "Industrial Internet", "Internet of Things", "Cloud Computing", "Big Data Analytics", "Artificial Intelligence", "Machine Learning", "Virtual Reality",
    "Process Optimization", "Lean Manufacturing", "Six Sigma", "Continuous Improvement", "Total Quality Management", "Supply Chain Management", "Project Management",
    "R&D Investment", "New Product Development", "Market Share", "Brand Value", "Intellectual Property", "Technical Standards", "Industry Norms",
    "Industrial Safety", "Workplace Health", "Emergency Plan", "Risk Assessment", "Compliance", "Insurance", "Liability Coverage",
    "International Trade", "Tariffs", "Export Volume", "Import Volume", "Exchange Rate Fluctuations", "Multinational Corporations", "Global Supply Chain",
    "Industrial Design", "Human-Machine Interaction", "User Experience", "Packaging Design", "Color Science", "Aesthetics", "Functional Design",
    "Energy Efficiency", "Renewable Energy", "Energy Conservation", "Clean Production", "Circular Economy", "Green Manufacturing", "Eco-Design",
    "Process Automation", "Quality Control", "Maintenance Schedule", "Product Lifecycle Management", "Supply Chain Integration", "Inventory Management", "Cost Reduction",
    "Productivity Metrics", "Operational Excellence", "Strategic Sourcing", "Demand Forecasting", "Capacity Planning", "Logistics Optimization", "Waste Management",
    "Innovation Index", "Technology Transfer", "Industry 4.0", "Smart Factory", "Digital Twin", "Augmented Reality", "Cybersecurity", "Data Privacy",
    "Market Analysis", "Competitive Landscape", "Regulatory Compliance", "Corporate Social Responsibility", "Sustainability Reporting", "Corporate Governance", "Ethical Sourcing",
    "Supply Chain Resilience", "Disaster Recovery", "Business Continuity", "Supply Chain Visibility", "Risk Mitigation", "Strategic Partnerships", "Joint Ventures",
    "Manufacturing Execution System", "Advanced Manufacturing", "Additive Manufacturing", "Precision Engineering", "Surface Engineering", "Heat Treatment", "Joining Techniques",
    "Material Handling", "Conveyor Systems", "Automated Storage and Retrieval Systems", "Warehouse Management", "Fleet Management", "Route Optimization", "Last Mile Delivery",
    "Process Simulation", "Virtual Commissioning", "Operator Training", "Remote Monitoring", "Predictive Maintenance", "Condition Monitoring", "Asset Management",
    "Supply Chain Analytics", "Demand-driven Planning", "Inventory Optimization", "Procurement Strategy", "Vendor Management", "Supply Chain Finance", "Supply Chain Collaboration",
    "Energy Management Systems", "Water Conservation", "Recycling Programs", "Waste Reduction Strategies", "Sustainable Packaging", "Green Building Materials", "Renewable Resource Use"
]
marketing_elements = [
    "GMV", "Customer Lifetime Value", "Click-Through Rate", "Conversion Rate", "Cost Per Click", "Cost Per Acquisition",
    "Return on Ad Spend", "Brand Awareness", "Brand Loyalty", "Market Share", "Customer Acquisition Cost",
    "Customer Retention Rate", "Customer Satisfaction Score", "Net Promoter Score", "Engagement Rate", "Social Media Reach",
    "Impression Share", "Search Engine Rankings", "Content Views", "Video Views", "Social Media Engagement", "Influencer Partnerships",
    "Target Audience", "Segmentation Strategy", "Positioning Statement", "Unique Selling Proposition", "Value Proposition", "Competitive Analysis",
    "Marketing Mix", "Product Development", "Price Strategy", "Promotion Strategy", "Place Strategy", "Advertising Campaign",
    "Public Relations", "Event Marketing", "Guerrilla Marketing", "Ambush Marketing", "Native Advertising", "Sponsorship Marketing",
    "Email Marketing", "Search Engine Marketing", "Social Media Marketing", "Content Marketing", "Mobile Marketing", "Affiliate Marketing",
    "Programmatic Advertising", "Retargeting", "Behavioral Targeting", "Geotargeting", "Demographic Targeting", "Psychographic Targeting",
    "Creative Strategy", "Copywriting", "Visual Design", "Video Production", "Audio Production", "Interactive Media", "User Experience",
    "Website Traffic", "Lead Generation", "Lead Nurturing", "Sales Funnel Optimization", "E-commerce Conversion", "Cart Abandonment",
    "Customer Journey Mapping", "Buyer Personas", "Brand Guidelines", "Logo Design", "Packaging Design", "Trade Dress",
    "Market Research", "Consumer Insights", "Focus Groups", "Surveys", "Data Analytics", "Consumer Behavior Analysis",
    "Market Segmentation", "Target Market Selection", "Product Positioning", "Brand Personality", "Brand Voice", "Brand Storytelling",
    "Content Strategy", "SEO Optimization", "Content Distribution", "Content Syndication", "Blog Posts", "Press Releases",
    "Whitepapers", "E-books", "Infographics", "Webinars", "Podcasts", "Virtual Events", "Trade Shows",
    "Sales Enablement", "Sales Training", "Sales Collateral", "Sales Forecasting", "Pipeline Management", "Sales Performance",
    "Customer Feedback", "Customer Reviews", "Testimonials", "Case Studies", "Referral Programs", "Loyalty Programs",
    "Cross-Selling", "Upselling", "Downselling", "Product Bundling", "Subscription Models", "Freemium Models",
    "Omnichannel Marketing", "Personalization", "Customer Data Platform", "Marketing Automation", "CRM Integration", "Customer Relationship Management",
    "Brand Ambassadors", "Community Building", "Social Listening", "Crisis Management", "Reputation Management", "Corporate Reputation",
    "Corporate Identity", "Corporate Communications", "Internal Marketing", "Employee Advocacy", "Employee Engagement", "Employer Branding",
    "Sustainability Marketing", "Green Marketing", "Eco-Friendly Practices", "Corporate Social Initiatives", "Social Responsibility", "Ethical Marketing"
]
natural_science_elements = [
    "Hydrogen", "Helium", "Lithium", "Beryllium", "Boron", "Carbon", "Nitrogen", "Oxygen", "Fluorine",
    "Neon", "Sodium", "Magnesium", "Aluminum", "Silicon", "Phosphorus", "Sulfur", "Chlorine", "Argon",
    "Potassium", "Calcium", "Scandium", "Titanium", "Vanadium", "Chromium", "Manganese", "Iron", "Cobalt",
    "Nickel", "Copper", "Zinc", "Gallium", "Germanium", "Arsenic", "Selenium", "Bromine", "Krypton",
    "Rubidium", "Strontium", "Yttrium", "Zirconium", "Niobium", "Molybdenum", "Technetium", "Ruthenium", "Rhodium",
    "Palladium", "Silver", "Cadmium", "Indium", "Tin", "Antimony", "Tellurium", "Iodine", "Xenon",
    "Cesium", "Barium", "Lanthanum", "Cerium", "Praseodymium", "Neodymium", "Promethium", "Samarium", "Europium",
    "Gadolinium", "Terbium", "Dysprosium", "Holmium", "Erbium", "Thulium", "Ytterbium", "Lutetium", "Hafnium",
    "Tantalum", "Tungsten", "Rhenium", "Osmium", "Iridium", "Platinum", "Gold", "Mercury", "Thallium",
    "Lead", "Bismuth", "Polonium", "Astatine", "Radon", "Francium", "Radium", "Actinium", "Thorium",
    "Protactinium", "Uranium", "Neptunium", "Plutonium", "Americium", "Curium", "Berkelium", "Californium", "Einsteinium",
    "Fermi", "Mendelevium", "Nobelium", "Lawrencium", "Rutherfordium", "Dubnium", "Seaborgium", "Bohrium", "Hassium",
    "Meitnerium", "Darmstadtium", "Roentgenium", "Copernicium", "Nihonium", "Flerovium", "Moscovium", "Livermorium", "Tennessine",
    "Oganesson", "Gravitational Force", "Electromagnetic Force", "Strong Nuclear Force", "Weak Nuclear Force", "Law of Conservation of Mass",
    "Law of Conservation of Energy", "Law of Universal Gravitation", "Law of Thermodynamics", "Law of Electromagnetism", "Quantum Mechanics",
    "Relativity", "Photosynthesis", "Cell Division", "Evolution", "Natural Selection", "Genetic Inheritance", "Biodiversity",
    "Ecological Balance", "Climate Change", "Plate Tectonics", "Geological Time Scale", "Fossil Record", "Continental Drift", "Volcanic Activity",
    "Earthquake", "Tsunami", "Hurricane", "Tornado", "Aurora Borealis", "Photospheric Eruptions", "Magnetic Field", "Solar Wind",
    "Cosmological Principle", "Big Bang Theory", "Dark Matter", "Dark Energy", "Black Hole", "Galaxy Formation", "Star Life Cycle",
    "Planetary Formation", "Stellar Evolution", "Cosmic Microwave Background", "Redshift", "Astronomical Unit", "Light Year", "Parsec",
    "Orbital Period", "Rotation Period", "Escape Velocity", "Tidal Forces", "Gravitational Lensing", "Event Horizon", "Singularity",
    "Supernova", "Pulsar", "Neutron Star", "White Dwarf", "Planetary Nebula", "Meteor Crater", "Impact Event", "Comet Encounter",
    "Astrobiology", "Exoplanets", "Interstellar Travel", "Space Exploration", "Space Colonization", "Artificial Gravity", "Space Debris",
    "Cosmic Radiation", "Microgravity", "Space Weather", "Space Anomaly", "Space-Time Continuum", "Quantum Entanglement", "Quantum Teleportation"
]

question_CG = {
    "IT":
        {
            "whether {} and {} is independent.",
            "Is {} independent of {}?",
            "Are {} and {} statistically independent?",
            "Does the occurrence of {} independent on {}, or vice versa?",
            "Can we assert {} and {} are independent, or are they related?",
            "Can we consider {} and {} as independent events?",
            "Do {} and {} independent and don't have any influence on each other?",
            "Is there no statistically correlation between {} and {}?",
            "test whether Are {} and {} statistically unrelated or dependent?",
            "Test the independence of {} and {}."
        },
    "CIT":
        {
            "whether {} and {} is independent under condition {}?",
            "Is {} independent of {} given condition {}?",
            "Are {} and {} statistically independent given the condition {}?",
            "Does the independence of {} and {} hold true under condition {}?",
            "Can we consider {} and {} as conditionally independent with respect to {}?",
            "Is the independence between {} and {} maintained given the condition {}?",
            "Does the occurrence of {} depend on {}, or vice versa, given condition {}?",
            "Are {} and {} conditionally independent with the presence of condition {}?",
            "Can we assume that {} and {} are independent given the condition {}?",
            "Is the independence of {} and {} upheld in the presence of condition {}?",
            "Does the independence between {} and {} persist under the condition {}?"
         },
    "MULTCIT" :
        {
            "whether {} and {} is independent under conditions : ",
            "Determine the independence of {} and {} given the following conditions : ",
            "Examine if {} and {} are independent under the specified conditions : ",
            "Assess the independence between {} and {} with the provided conditions : ",
            "Investigate whether {} and {} exhibit independence given the outlined conditions : ",
            "Explore the independence of {} and {} under the given circumstances : ",
            "Ascertain if there is independence between {} and {} given the stated conditions : ",
            "Check for independence between {} and {} based on the conditions described : ",
            "Verify the independence status of {} and {} under the listed conditions : ",
            "Evaluate the independence of {} and {} under the mentioned conditions : ",
            "Examine whether {} and {} are independent, considering the provided conditions : "
        },
    "CAUSE" :
        {
            "whether {} directly cause {}.",
            "Assess if {} has a direct causal impact on {}.",
            "Examine the direct causation relationship.if {} directly cause {}?",
            "Investigate whether {} directly influences {}.",
            "Evaluate if there exists the direct causal connection from {} to {}.",
            "Scrutinize if {} leads to a direct causation of {}.",
            "Determine whether {} is a direct cause of {}.",
            "Assess if there is the direct causal link of {} to {}.",
            "Verify if {} directly results in the causation of {}."
        },
    "Has-Collider" :
        {
            "Whether there exists at least one collider (i.e., common effect) of {} and {}",
            "Determine if there is at least one common effect (collider) of both {} and {}.",
            "Assess the presence of a shared outcome, serving as a collider, for variables {} and {}.",
            "Examine the potential existence of a shared consequence as a collider for {} and {}.",
            "Evaluate if {} and {} share a common effect (collider).",
            "Analyze the presence of a common outcome serving as a collider for {} and {}.",
            "Verify if there exists a shared effect, acting as a collider, for both {} and {}.",
            "Explore whether a common consequence is a collider for variables {} and {}.",
            "Assess the existence of at least one common effect (collider) between {} and {}."
        },
    "Has-Confounder" :
        {
            "There exists at least one confounder (i.e., common cause) of {} and {}.",
            "Confirm the presence of at least one common cause (confounder) influencing both {} and {}.",
            "Verify whether there exists a shared factor, acting as a confounder, for variables {} and {}.",
            "Examine the potential existence of a common cause (confounder) impacting both {} and {}.",
            "Assess if {} and {} share at least one confounding factor (common cause).",
            "Scrutinize the presence of a shared influencing factor, serving as a confounder, for {} and {}.",
            "Investigate whether there is at least one confounder affecting both {} and {}.",
            "Analyze the potential impact of a common cause (confounder) on variables {} and {}.",
            "Verify the presence of a shared influencing factor, acting as a confounder, for {} and {}.",
            "Explore whether a common factor is a confounder for variables {} and {}.",
            "Evaluate the existence of at least one confounder (common cause) between {} and {}."
    },
    "CAUSALKG" :
        {
            "please generate causal graph of the input tabular data.",
            "Produce a causal graph representing the relationships within the given tabular data.",
            "Generate a directed graph that illustrates the causal connections inherent in the provided tabular dataset.",
            "Create a graphical model depicting the causality among variables in the input tabular data.",
            "Construct a causal diagram illustrating the interdependencies among the variables in the tabular dataset.",
            "Formulate a graph that visually represents the cause-and-effect relationships present in the input tabular information.",
            "Develop a graphical representation outlining the causal structure of the tabular data.",
            "Build a directed acyclic graph (DAG) that reflects the causal influences within the input tabular dataset.",
            "Establish a graphical model showcasing the causal links between variables derived from the tabular data.",
            "Design a causal graph that visually captures the cause-and-effect relationships inherent in the tabular information.",
            "Construct a directed graph that visually displays the causal pathways within the given tabular dataset."
        },
    "PARTIAL_CG":
        {
            "Please generate a partial causal diagram for some of the following variables that interest me : ",
            "Generate a subset of a causal diagram for the variables of interest : ",
            "Create a partial graphical model illustrating causal relationships among selected variables : ",
            "Develop a restricted causal graph focusing on specific variables from the given set : ",
            "Formulate a partial directed acyclic graph (DAG) depicting causal connections for chosen variables : ",
            "Construct a limited causal diagram featuring only the variables of interest : ",
            "Produce a subsection of a graphical model, emphasizing the causal links within the selected variables : ",
            "Build a causal graph subset, emphasizing relationships among the variables you find intriguing : ",
            "Develop a focused causal diagram, highlighting causal connections for the specified variables : ",
            "Form a segment of a directed graph that visually represents causal relationships among chosen variables : ",
            "Create a restricted causal network, showcasing the partial causal influences among the variables of interest : "
        },
    "ATE":{
        "calculate the Average Treatment Effect (ATE) of a continuous treatment variable  {T} on an outcome variable {Y}, given that the treatment {T} change from {T0} to {T1}."
    }
}

print(len(medical_elements))

prompt_template = '''
##Requirements: Suppose you are a statistician and need to perform causal analysis on data. You need to use your imagination to compile a reasonable scene description based on the following elements, and finally ask a question Q: " {} ". The scenario description needs to be related to the problem and form a paragraph together with the problem. This output must end up with the question format, either directly end up with the  question Q or the equivalent of the question Q. Below are all the elements you need to use to describe the scenario (including those involved in the  question Q). Elements don't exist in variables listed below are not allowed.

##element:[{}]

##Output:
'''
gpt = ChatGPT()
counter = 0
index = 0
ind = 0
skip = 0
file_name = "./dataset_ate.json"
for node_num in range(3,11):
    item = "calculate the Average Treatment Effect (ATE) of a continuous treatment variable  {T} on an outcome variable {Y}, given that the treatment {T} change from {T0} to {T1}."
    sampled_elements = random.sample(medical_elements, node_num)
    interest = random.sample(sampled_elements, 2)
    T1 = 0.2
    T0 = 0.5
    while True:
        T1 = round(random.random()*2 + -1,2)
        T0 = round(random.random()*2 + -1,2)
        if T1 != T0:
            break
    question = item.format(T=interest[0], Y=interest[1],T0=T0,T1=T1)
    prompt = prompt_template.format(question, ','.join(sampled_elements))
    resp = gpt.call(prompt)
    print(resp)
    one_item = {'node num': node_num, 'question_type': "ATE", 'interest': interest, 'variables': sampled_elements,'T0':T0,'T1':T1,
                'text': resp}
    with open(file_name, 'a+') as f:
        json.dump(one_item, f, ensure_ascii=False)
        f.write('\n')

# for node_num in range(3,11):
#     for Q in question_CG.keys():
#         for item in question_CG[Q]:
#             if ind < skip:
#                 ind += 1
#                 continue
#             if counter < index:
#                 counter += 1
#                 continue
#
#             if Q in ["IT","CAUSE","Has-Collider","Has-Confounder"]:
#                 sampled_elements = random.sample(medical_elements, node_num)
#                 interest = random.sample(sampled_elements, 2)
#                 question = item.format(interest[0],interest[1])
#                 prompt = prompt_template.format(question,','.join(sampled_elements))
#                 resp = gpt.call(prompt)
#                 print(resp)
#                 one_item = {'node num':node_num,'question_type':Q,'interest':interest,'variables':sampled_elements,'text': resp }
#                 with open(file_name,'a+') as f:
#                     json.dump(one_item, f, ensure_ascii=False)
#                     f.write('\n')
#             if Q in ["CIT"]:
#                 sampled_elements = random.sample(medical_elements, node_num)
#                 interest = random.sample(sampled_elements, 3)
#                 question = item.format(interest[0],interest[1],interest[2])
#                 prompt = prompt_template.format(question,','.join(sampled_elements))
#                 resp = gpt.call(prompt)
#                 print(resp)
#                 one_item = {'node num':node_num,'question_type':Q,'interest':interest,'variables':sampled_elements,'text': resp }
#                 with open(file_name,'a+') as f:
#                     json.dump(one_item, f, ensure_ascii=False)
#                     f.write('\n')
#             if Q in ["MULTCIT"]:
#                 sampled_elements = random.sample(medical_elements, node_num)
#                 rand_num = random.randint(3,node_num)
#                 interest = random.sample(sampled_elements, rand_num)
#                 question = item.format(interest[0],interest[1]) + ','.join(interest[2:])
#                 prompt = prompt_template.format(question,','.join(sampled_elements))
#                 resp = gpt.call(prompt)
#                 print(resp)
#                 one_item = {'node num':node_num,'question_type':Q,'interest':interest,'variables':sampled_elements,'text': resp }
#                 with open(file_name,'a+') as f:
#                     json.dump(one_item, f, ensure_ascii=False)
#                     f.write('\n')
#             if Q in ["CAUSALKG"]:
#                 sampled_elements = random.sample(medical_elements, node_num)
#                 question = item
#                 prompt = prompt_template.format(question,','.join(sampled_elements))
#                 resp = gpt.call(prompt)
#                 print(resp)
#                 one_item = {'node num':node_num,'question_type':Q,'interest':[],'variables':sampled_elements,'text': resp }
#                 with open(file_name,'a+') as f:
#                     json.dump(one_item, f, ensure_ascii=False)
#                     f.write('\n')
#             if Q in ["PARTIAL_CG"]:
#                 if node_num==3:
#                     continue
#                 sampled_elements = random.sample(medical_elements, node_num)
#                 rand_num = random.randint(3,node_num-1)
#                 interest = random.sample(sampled_elements, rand_num)
#                 question = item + ','.join(interest)
#                 prompt = prompt_template.format(question,','.join(sampled_elements))
#                 resp = gpt.call(prompt)
#                 print(resp)
#                 one_item = {'node num':node_num,'question_type':Q,'interest':interest,'variables':sampled_elements,'text': resp }
#                 with open(file_name,'a+') as f:
#                     json.dump(one_item, f, ensure_ascii=False)
#                     f.write('\n')
#         # sampled_elements = random.sample(medical_elements, node_num)
#         # print(sampled_elements)
#         # print(prompt.format(','.join(sampled_elements)))
#

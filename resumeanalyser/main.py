from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm=OllamaLLM(model="llama3.2")
p_=""" You are Resume analyser that capable of giving the desired ouput for the give job description.
Based on analysis you should give the output 
1. Match score        → "72% match"
2. Missing skills     → "You lack: Kubernetes, GraphQL"
3. Strong points      → "You have: Python, RAG, LangChain"
4. Recommendations    → "Add these to your resume: ..."

The Job description as follows.
<Job description starts>
{job_description}
<Job description end>

<Resume starts>
{resume}
<Resume ends>
"""
prompt=PromptTemplate(template=p_,input_variables=["job_description","resume"])

job_descrption_file_open=open("jd.txt")
jd_file_read=job_descrption_file_open.read()
job_descrption_file_open.close()

resume_file_open=open("resume.txt")
resume_file_read=resume_file_open.read()
resume_file_open.close()

chain=(  prompt | llm | StrOutputParser())

print(chain.invoke({"job_description":jd_file_read,"resume":resume_file_read}))
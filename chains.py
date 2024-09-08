
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException

load_dotenv()
os.getenv("GROQ_API_KEY")

class Chain:
    def __init__(self):
        self.llm = ChatGroq(
    temperature=0,
    groq_api_key="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",
    model_name='llama-3.1-70b-versatile')

    def extract_jobs(self, page_data):
        prompt_extract = PromptTemplate.from_template(
        """
        ###SCRAPED TEXT FROM WEBSITE###
        {page_data}
        ### INSTRUCTIONS:
        The scraped text is from career's page of website.
        your job is to extract the job posting and return them in JSON format containing
        following keys: 'role', 'experience', 'skills', 'description'.
        Only return a valid JSON.
        ###VALID JSON (NO PREAMBLE):
        """
        )
        chain_extract = prompt_extract | self.llm
        result = chain_extract.invoke(input={'page_data':page_data})
        try:
            json_parser= JsonOutputParser()
            result = json_parser.parse(result.content)
        except OutputParserException:
                raise OutputParserException("Context Too Big, Unable to Parse")
        return result if isinstance(result, list) else [result]

    def write_email(self, job, link):
        prompt_email= PromptTemplate.from_template(
            """
            ###JOB DESCRIPTION###
            {job_description}
            ### INSTRUCTION:
            You are M.Sameet Javed, a business development executive at Datics AI. Datics AI is an AI & Software Consulting company dedicated to facilitating
            the seamless integration of business processes through automated tools.  
            Your job is to write a cold email to the client regarding the job mentioned above describing the capability of Datics AI 
            in fulfilling their needs.
            Also add the most relevant ones from the following links to showcase Datics AI's portfolio: {link_list}
            Remember you are M.Sameet Javed, BDE at Datics. 
            Do not provide a preamble.
            ### EMAIL (NO PREAMBLE):
            """
        )
        chain_email = prompt_email | self.llm
        res = chain_email.invoke({'job_description': str(job), 'link_list': link})
        return res.content    


if __name__ == "__main__":
     os.getenv("GROQ_API_KEY")




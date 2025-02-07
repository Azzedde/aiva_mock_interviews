from openai import OpenAI

class AIInterviewer:

    def __init__(self, base_url, api_key, model):
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.client = OpenAI(base_url, api_key)

    def init_stream_response(self):

        user_prompt =  """Ask an insightful, open-ended question about their AI experience
        based on their CV and introduction.

        Candidate Introduction: {user_response}
        CV: {cv}

        Ask a probing question that reveals the candidate's depth of AI knowledge and experience."""

        formatted_user_prompt = user_prompt.format()

        response = self.client.chat.completions.create(
        self.model,
        messages=[
            {"role": "system", "content": "You are a professional talent acquisition specialist interviewing a candidate for an AI role."},
            {"role": "user", "content": "exemple d'une question"},
            {"role": "assistant", "content": "exemple d'une reponse"},
            {"role": "user", "content": formatted_user_prompt}
        ],
        stream=True,
        )
        with response as stream:
            for chunk in stream:
                yield chunk

    def stream_next_question(self, precedent_response):
        
        user_prompt = """Ask an insightful, open-ended question about their AI experience
        based on their CV and introduction.
        
        Precedent_response : {precedent_reponse}"""
        
        formatted_user_prompt = user_prompt.format(precedent_response)

        response = self.client.chat.completions.create(
        self.model,
        messages=[
            {"role": "system", "content": "You are a professional talent acquisition specialist interviewing a candidate for an AI role."},
            {"role": "user", "content": "exemple d'une question"},
            {"role": "assistant", "content": "exemple d'une reponse"},
            {"role": "user", "content": formatted_user_prompt}
        ],
        stream=True,
        )
        with response as stream:
            for chunk in stream:
                yield chunk
        return
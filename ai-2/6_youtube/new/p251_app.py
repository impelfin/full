import openai
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter
import textwrap

def get_video_id(video_url):
    video_id = video_url.split("v=")[1][:11]

    return video_id

video_url = "https://www.youtube.com/watch?v=pSJrML-TTmI" 
video_id = get_video_id(video_url)

try:
    transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
    
    try:
        transcript_obj = transcript_list.find_transcript(['ko'])
    except NoTranscriptFound:
        transcript_obj = transcript_list.find_generated_transcript(['ko'])

    transcript = transcript_obj.fetch()

    text_formatter = TextFormatter()
    text_formatted = text_formatter.format_transcript(transcript)
    text_info = text_formatted.replace('\n', " ")
    
except TranscriptsDisabled:
    print("이 영상은 자막이 비활성화되어 있습니다.")
except NoTranscriptFound:
    print("이 영상에는 한국어 자막(수동/자동 생성 포함)이 없습니다.")
except Exception as e:
    print(f"알 수 없는 오류 발생: {e}")

def answer_from_given_info(question_info, prompt):
    user_content = f"{prompt} 다음 내용을 바탕으로 질문에 답해 줘. {question_info}"

    message = [
        { "role": "user", "content": user_content }
    ]

    response = openai.ChatCompletion.create(
        model = "gpt-3.5-turbo",
        messages = message,
        max_tokens = 500,
        stop = ["."],
        temperature = 0.2,
    )

    return response['choices'][0]['message']['content']

question_info = text_info
prompt = "허준이 교수가 받은 상은 무엇인가요?"
print(prompt)
response = answer_from_given_info(question_info, prompt)
print(response)
print('-' * 70)

question_info = text_info
prompt = "허준이 교수는 어느 대학 교수인가요?"
print(prompt)
response = answer_from_given_info(question_info, prompt)
print(response)
print('-' * 70)
    
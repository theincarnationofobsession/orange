# Installations for different important libraries
!pip install llama_index
!pip install --upgrade -q accelerate bitsandbytes
!pip install git+https://github.com/huggingface/transformers.git

# Importing the libaries
from llama_index import VectorStoreIndex, SimpleDirectoryReader
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
from transformers import pipeline
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from llama_index.llms import OpenAI
import cv2
import torch
from PIL import Image
import os

quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)

def extract_keyframes(video_path, output_folder, num_keyframes=1):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Unable to open video file.")
        return
    # Calculate total number of frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = total_frames // (num_keyframes + 1)
    frame_count = 0
    keyframe_count = 0

    # Read the first frame
    ret, frame = cap.read()

    while ret:
        if frame_count == interval * (keyframe_count + 1):
            keyframe_path = "/keyframe.jpg"
            cv2.imwrite(keyframe_path, frame)
            keyframe_count += 1
            if keyframe_count >= num_keyframes:
                break

        ret, frame = cap.read()
        frame_count += 1

    # Release the video capture object
    cap.release()


model_id = "llava-hf/llava-1.5-7b-hf"
processor = AutoProcessor.from_pretrained(model_id)
model = LlavaForConditionalGeneration.from_pretrained(model_id, quantization_config=quantization_config, device_map="auto")

pipe = pipeline("image-to-text", model=model_id, model_kwargs={"quantization_config": quantization_config})

def inference(video_path):
  extract_keyframes(video_path, '/keyframe.jpg', num_keyframes=1)
  image = Image.open('/keyframe.jpg')

  max_new_tokens = 200
  # audio_to_text = "Oh my God, he’s lost it. He’s totally lost it."
  prompt = f"USER: <image>\nTell me the emotion of the person in the image from neutral, joy, anger, surprise, sadness, fear, disgust?\nASSISTANT:"

  outputs = pipe(image, prompt=prompt, generate_kwargs={"max_new_tokens": 200})

  return outputs[0]["generated_text"].split("ASSISTANT:")[-1]

def generate(query, emotion):
    os.environ["OPENAI_API_KEY"] = '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'
    documents = SimpleDirectoryReader("/content/drive/MyDrive/Llama_Index/Summarization/").load_data()
    index = VectorStoreIndex.from_documents(documents)

    query_engine = index.as_query_engine(similarity_top_k=5)
    response = query_engine.query("What exactly is the disease")
    disease_info = str(response)
    
    model = OpenAI(model="gpt-4-vision-preview")
    corpus_prompt = "We know that the patient is facing '{}'. Tell more about the condition faced by the patient in a concise, detailed manner.".format(disease_info)
    context_response = query_engine.query(corpus_prompt)
    
    query_engine = index.as_query_engine(similarity_top_k=5)
    response = query_engine.query(query)
    actual_query = str(response)
    
    final_prompt = "Given the query: {}, the details of the condition faced by the patient {}, and the user's emotion: {}, generate only the consultant's reply and nothing else:".format(actual_query,context_response,emotion)
    query_engine = index.as_query_engine(llm=model)
    output = query_engine.query(final_prompt)
    
    return output

emoticon = inference("/content/video.mp4")
generate("What are the side effects", emoticon)
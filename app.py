import validators,streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader,UnstructuredURLLoader


st.set_page_config(page_title="LangChain: Summarize Text From YT or Website", page_icon="🦜")
st.title("🦜 LangChain: Summarize Text From YT or Website")
st.subheader('Summarize URL')

with st.sidebar:
    groq_api_key=st.text_input("Groq API Key",value="",type="password")
    
generic_url=st.text_input("URL",label_visibility="collapsed") # it means the input is of type URL, label means the text that appears above the input box and visibility means whether the label is visible or not


prompt_template="""
Provide a summary of the following content in 300 words:
Content:{text}

"""
prompt=PromptTemplate(template=prompt_template,input_variables=["text"])

if st.button("Summarize the Content from YT or Website"):
    ## Validate all the inputs
    if not groq_api_key.strip() or not generic_url.strip(): # if groq_api_key is empty or generic_url is empty then raise an error ,strip() is used to remove the spaces between the url
        st.error("Please provide the information to get started")
    elif not validators.url(generic_url): # if the url is not valid then raise an error, validators.url() is used to validate the url, validate means to check whether the url is valid or not
        st.error("Please enter a valid Url. It can may be a YT video utl or website url")

    else: # if all the inputs are valid then
        try:
            with st.spinner("Waiting..."): # show the spinner while the code is running
                ## loading the Groq API model
                llm =ChatGroq(model="Llama-3.1-8b-Instant", groq_api_key=groq_api_key)
                ## loading the website or yt video data
                if "youtube.com" in generic_url or "youtu.be" in generic_url:
                        # Clean the YouTube URL and extract video ID
                    if "youtu.be/" in generic_url:
                            video_id = generic_url.split("youtu.be/")[1].split("?")[0]
                            clean_url = f"https://www.youtube.com/watch?v={video_id}"
                    else:
                            clean_url = generic_url.split("&")[0]  # Remove extra parameters
                        
                    st.info(f"Loading YouTube video: {clean_url}")
                    loader = YoutubeLoader.from_youtube_url(
                            clean_url, 
                            add_video_info=False,  # Set to False to avoid metadata issues
                            language=["en", "en-US"]  # Specify language preferences
                        ) # if the url contains youtube.com then load the video and add the video info
                else:
                    loader=UnstructuredURLLoader(urls=[generic_url],ssl_verify=False,
                                                 headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"})
                    # if the url does not contain youtube.com then load the url with ssl_verify=False i.e. do not verify the ssl certificate as some websites may not have a valid ssl
                    # certificate, headers is used to set the user agent to make the request look like it is coming from a browser as some websites may block requests that do not have a user agent set
                docs=loader.load()

                ## Chain For Summarization
                chain=load_summarize_chain(llm,chain_type="stuff",prompt=prompt) # create a chain for summarization using the llm and prompt, chain_type="stuff" means that the chain 
                # will use the stuff method to summarize the text
                output_summary=chain.run(docs)

                st.success(output_summary) # display the output summary in the app
        except Exception as e:
            st.error("🚨 Something went wrong during summarization.")
            st.code(str(e), language="python")  
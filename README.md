# NIC-English-Api
API that takes audio input and gives its actual transcription along with phoneme 

# To run the server
pip install -r requirements.txt
uvicorn main:app --reload

# To test the server 
curl -X POST http://localhost:8000/transcribe/file -F "file=@*/path/filename.wav"

# Example:
curl -X POST http://localhost:8000/transcribe/file -F "file=@speech_orig.wav"

# Example Output:
{
    "text":"the boch canoe slit on the smooth planks blew the sheet to the dark blue background it's easy to tell a depth of a well four hours of steady work faced us",
    "success":true,
    "phonetic":"ðə bɑk kəˈnu slɪt ɔn ðə smuð plæŋks blu ðə ʃit tɪ ðə dɑrk blu ˈbækˌgraʊnd ɪts ˈizi tɪ tɛl ə dɛpθ əv ə wɛl fɔr aʊərz əv ˈstɛdi wərk feɪst ˈjuˈɛs",
    "error":null}

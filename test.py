#采用AI-HUB方式
#pip install ai-hub
import json
from ai_hub import inferServer
class AIHubInfer(inferServer):
    def __init__(self, model):
        super().__init__(model)
    #数据前处理
    def pre_process(self, data):
        print("pre_process")
        data = data.get_data()
        # json process
        json_data = json.loads(data.decode('utf-8'))

        # 这个示例的是将接收天池服务器的流评测内容，将图片写到本地的submit目录：
        if os.path.exists('submit'):
            shutil.rmtree('submit')
            print ('Delete submit folder')
            os.makedirs('submit')
            print ('Create submit folder')
        for category in ['Wind', 'Precip', 'Radar']:
            category_path = os.path.join('submit', category )
            os.makedirs(category_path)
            for png_name, base64_string in json_data[category].items():
	      # 请选手注意，在天池的流评测服务器环境下，此处获取到的base64_string其实是一个list结构，非str，选手需要显现得取第一个元素才能得到正确的图片base64编码内容。
                base64_string = base64_string[0]
                file_name = os.path.join(category_path, png_name)
                img = Image.open(BytesIO(base64.urlsafe_b64decode(base64_string)))
                img.save(file_name)
   
    #数据后处理，如无，可空缺
    def post_process(self, predict_data):
        elem = {}
        # 读取模型预测好的文件，按照如下格式返回, 本例为了说明问题，直接去读输入文件的第一条作为输出结果：
        elem['Wind'] = {}
        elem['Radar'] = {}
        elem['Precip'] = {}
        for category in ['Wind', 'Precip', 'Radar']:
            mock_file_path = os.path.join('submit', category, f"{category.lower()}_001.png")
            print ('mock_file_path: ', mock_file_path)
            binary_content = open(mock_file_path, 'rb').read()
            base64_bytes = base64.b64encode(binary_content)
            base64_string = base64_bytes.decode('utf-8')
            for idx in range(1, 21):
                file_name = os.path.join('', f"{category.lower()}_{idx:03d}.png")
                print ('Post_process: ', category, file_name)
	       # 此处选手直接写入base64字符串即可，不用转成list结构。
                elem[category][file_name] = base64_string

        # 返回json格式
        return json.dumps(elem)
    
if __name__ == "__main__":
	mymodel = lambda x: x * 2
	my_infer = AIHubInfer(mymodel)
	my_infer.run(debuge=True)    
#ai-hub使用方向可参见：https://github.com/gaoxiaos/AI_HUB/blob/main/ai_hub/test.py

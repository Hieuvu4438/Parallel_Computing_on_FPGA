import requests
import json
import time

# Thay thế bằng API key của bạn
API_KEY = "AIzaSyB0rmqo9lLrZxv6UgcuvjDUTsaokHN8gj0"

def test_gemini_api_quota(api_key):
    print(f"Đang kiểm tra quota cho API key: {api_key[:10]}... ")
    print("-" * 50)
    
    # Bước 1: Kiểm tra tính hợp lệ cơ bản của Key và lấy danh sách Model hỗ trợ Generate
    print("[1] Kiểm tra tính hợp lệ cơ bản (Get Models)...")
    models_url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
    target_model = None
    try:
        response_models = requests.get(models_url)
        if response_models.status_code == 200:
            print(" ✅ API Key hợp lệ.")
            
            # Tìm model đầu tiên hỗ trợ 'generateContent'
            for item in response_models.json().get('models', []):
                if 'generateContent' in item.get('supportedGenerationMethods', []):
                    # extract the model name like 'models/gemini-2.5-flash' -> 'gemini-2.5-flash' (already has models/ prefix, but we will use the full name format)
                    target_model = item['name'].split('/')[-1]
                    break
            
            if not target_model:
                print(" ❌ THEO NHƯ API: Không tìm thấy Model nào hỗ trợ text generation.")
                return
            else:
                print(f" ⚙️ Tự động chọn model: {target_model}")
                
        elif response_models.status_code == 400:
            print(" ❌ LỖI 400: API Key không đúng định dạng.")
            print(" Chi tiết:", response_models.json().get("error", {}).get("message"))
            return
        elif response_models.status_code == 403:
            print(" ❌ LỖI 403: API Key KHÔNG CÓ QUYỀN TRUY CẬP (Forbidden).")
            print(" Chi tiết:", response_models.json().get("error", {}).get("message"))
            return
        else:
            print(f" ❌ LỖI {response_models.status_code}: {response_models.text}")
            return
    except requests.exceptions.RequestException as e:
        print(f" ❌ LỖI KẾT NỐI: Lỗi khi truy cập Google API. Chi tiết: {e}")
        return

    # Bước 2: Kiểm tra Quota/Billing thông qua việc Request Generate Text
    # Khi gọi generateContent, model sẽ tính token và trừ quota. Nếu hết quota sẽ báo 429 hoặc 403.
    print(f"\n[2] Kiểm tra hạn mức khả dụng (Quota) qua tác vụ tạo văn bản trên {target_model}...")
    
    generate_url = f"https://generativelanguage.googleapis.com/v1beta/models/{target_model}:generateContent?key={api_key}"
    
    headers = {
        'Content-Type': 'application/json'
    }
    
    data = {
        "contents": [{
            "parts": [{"text": "Hello, this is a quota check. Please reply with \"OK\"."}]
        }]
    }
    
    try:
        start_time = time.time()
        response_gen = requests.post(generate_url, headers=headers, json=data)
        elapsed_time = time.time() - start_time
        
        if response_gen.status_code == 200:
            print(" ✅ THÀNH CÔNG: API Key NÀY VẪN CÒN QUOTA và đang hoạt động tốt!")
            res_json = response_gen.json()
            
            try:
                text = res_json['candidates'][0]['content']['parts'][0]['text'].strip()
                print(f" 📝 Phản hồi model ({elapsed_time:.2f}s): {text}")
            except (KeyError, IndexError):
                print(" ⚠️ Phản hồi thành công nhưng không parse được text.")
            
            # In ra thông tin Token Usage
            if 'usageMetadata' in res_json:
                usage = res_json['usageMetadata']
                print(f" 📊 Token Usage - Bạn vừa dùng: Prompt ({usage.get('promptTokenCount', 0)}), Response ({usage.get('candidatesTokenCount', 0)}), Total ({usage.get('totalTokenCount', 0)})")
                
        elif response_gen.status_code == 429:
            print(" ❌ LỖI 429 (Too Many Requests / Quota Exceeded):")
            print(" ⚠️ API Key này ĐÃ HẾT QUOTA (hạn mức miễn phí hoặc trả phí) hoặc vượt quá giới hạn rate limit.")
            print(" Chi tiết từ server:", response_gen.json().get("error", {}).get("message"))
            
        elif response_gen.status_code == 403:
            print(" ❌ LỖI 403: Không có quyền truy cập Generate API.")
            print(" ⚠️ Key có thể bị khóa, chưa bật quyền Billing, hoặc bị cấm do quốc gia/khu vực.")
            print(" Chi tiết từ server:", response_gen.json().get("error", {}).get("message"))
            
        elif response_gen.status_code == 400:
            print(" ❌ LỖI 400: Yêu cầu không hợp lệ (Bad Request).")
            print(" Chi tiết từ server:", response_gen.json().get("error", {}).get("message"))
            
        else:
            print(f" ❌ LỖI {response_gen.status_code}: {response_gen.text}")
            
    except requests.exceptions.RequestException as e:
        print(f" ❌ LỖI KẾT NỐI: Không kết nối được tới endpoint generate. Chi tiết: {e}")

if __name__ == "__main__":
    test_gemini_api_quota(API_KEY)

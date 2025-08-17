import cv2
import numpy as np
import mss
import tempfile
import os
from paddleocr import PaddleOCR
from deep_translator import GoogleTranslator

# Khởi tạo OCR + Translator
ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False
)
translator = GoogleTranslator(source="auto", target="vi")

# Vùng màn hình cần OCR (full HD screen 1920x1080, bạn chỉnh lại nếu khác)
monitor = {"top": 0, "left": 0, "width": 1920, "height": 1080}

def overlay_translate():
    with mss.mss() as sct:
        while True:
            try:
                # Chụp màn hình
                screenshot = sct.grab(monitor)
                
                # Lưu screenshot tạm thời để OCR đọc
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                    temp_path = temp_file.name
                    mss.tools.to_png(screenshot.rgb, screenshot.size, output=temp_path)
                
                # OCR predict từ file
                results = ocr.predict(temp_path)
                
                # Xóa file tạm
                os.unlink(temp_path)
                
                # Chuyển screenshot thành frame để hiển thị
                frame = np.array(screenshot)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                
                # Xử lý kết quả OCR
                for res in results:
                    # Lấy dữ liệu từ result object
                    if hasattr(res, 'res') and res.res:
                        ocr_data = res.res
                        texts = ocr_data.get('rec_texts', [])
                        scores = ocr_data.get('rec_scores', [])
                        boxes = ocr_data.get('rec_boxes', [])
                        
                        # Vẽ box + dịch
                        for text, confidence, box in zip(texts, scores, boxes):
                            if not text.strip() or confidence < 0.5:  # Lọc text có độ tin cậy thấp
                                continue
                            
                            try:
                                # Dịch text
                                translated = translator.translate(text)
                                
                                # Lấy tọa độ từ rec_boxes - format [x1, y1, x2, y2]
                                x1, y1, x2, y2 = map(int, box)
                                
                                # Vẽ nền mờ cho dễ đọc
                                overlay = frame.copy()
                                cv2.rectangle(overlay, (x1, y1-25), (x2, y1), (0, 0, 0), -1)
                                cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
                                
                                # Vẽ text gốc (màu trắng)
                                cv2.putText(frame, text, (x1, max(25, y1-15)),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                                
                                # Vẽ text dịch (màu xanh lá)
                                cv2.putText(frame, translated, (x1, max(25, y1-5)),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                                
                                # Debug console
                                print(f"[OCR] {text} ({confidence:.2f}) -> [VI] {translated}")
                            
                            except Exception as translate_error:
                                print(f"Translation error: {translate_error}")
                                continue
                
                # Hiển thị overlay
                cv2.imshow("AI Overlay Translate", frame)
                
                # Thoát bằng phím q
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                    
            except Exception as e:
                print(f"Error in main loop: {e}")
                continue
        
        cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Starting AI Overlay Translator...")
    print("Press 'q' to quit")
    overlay_translate()
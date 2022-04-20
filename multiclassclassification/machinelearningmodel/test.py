# type: ignore

import requests

data = "गोरखा र नुवाकोटमा भूकम्पबाट क्षतिग्रस्त पचास हजार घरको पुनर्निर्माणका लागि भारत सरकारले १ करोड ६२ लाख अमेरिकी डलर (१ अर्ब ६७ करोड ७० लाख रुपैयाँ) अनुदान सहयोग गर्ने भएको छ ।"


r = requests.post(f'http://localhost:8000/input/', json={"text": data}).json()

print(r)

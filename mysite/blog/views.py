from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
import base64
from django.core.files.base import ContentFile

# Create your views here.
@csrf_exempt
def post_list(request):
    result = list(request.POST.keys())
    if(request.method == 'POST'):

        data = result[0][22:]
        # missing_padding = len(data) % 4
        # if missing_padding != 0:
        #     data += b'='*(4-missing_padding)
        #print(base64.b64decode(data))

        # pad =  len(data)%4
        #
        # a = "="*pad
        # data+=a
        # print(data)
        #data += "=" * ((4 - len(data) % 4) % 4)
        # img = base64.b64decode(data)

        img = base64.b64decode(data)

        fh = open("imageToSave.png", "wb")
        fh.write(img)
        fh.close()
        print("save success!")

    return render(request, 'blog/post_list.html', {})

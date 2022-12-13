import mahotas as mh
import streamlit as st
import numpy as np
import pickle
from io import BytesIO
from keras.models import model_from_json

st.title('Final Project AB - Kelompok 9')

filename = 'TB-model/finalized_model.sav'
model = pickle.load(open(filename, 'rb'))

STYLE = """
<style>
img {
    max-width: 100%;
}
</style>
"""


class FileUpload(object):

    def __init__(self):
        self.fileTypes = ["png", "jpg", "jpeg"]

    def run(self):
        """
        Upload File on Streamlit Code
        :return:
        """
        st.subheader("Predict Tuberculosis Using MLP")
        st.markdown(STYLE, unsafe_allow_html=True)
        file = st.file_uploader(
            "Upload your chest x-ray image:", type=self.fileTypes)
        show_file = st.empty()
        if not file:
            show_file.info("Please upload a file of type: " +
                           ", ".join(["png", "jpg", "jpeg"]))
            return
        content = file.getvalue()
        if isinstance(file, BytesIO):
            lab = {'Tuberculosis': 0, 'Normal': 1}
            IMM_SIZE = 224
            show_file.image(file)
            image = mh.imread(file)
            if len(image.shape) > 2:
                # resize of RGB and png images
                image = mh.resize_to(
                    image, [IMM_SIZE, IMM_SIZE, image.shape[2]])
            else:
                # resize of grey images
                image = mh.resize_to(image, [IMM_SIZE, IMM_SIZE])
            if len(image.shape) > 2:
                # change of colormap of images alpha chanel delete
                image = mh.colors.rgb2grey(image[:, :, :3], dtype=np.uint8)

            inputs = [mh.features.haralick(image).ravel()]
            prediction = model.predict(inputs)

            prediction = model.predict(inputs)
            st.code(
                f"Prediction: {np.squeeze(prediction, -1)}")

        file.close()


if __name__ == "__main__":
    helper = FileUpload()
    helper.run()

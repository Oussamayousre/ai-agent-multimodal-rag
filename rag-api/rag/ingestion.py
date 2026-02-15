import langchain
from langchain_community.document_loaders import PyPDFLoader,TextLoader
from pdf2image import convert_from_path


class DocumentsLoader : 
    """
    this class contains all type of files loading , from text loader to image+text+tables loading
    """
    def __init__(self, path , chunk_size = 300 , chunk_overlap = 20 ,img = ""):
        """
        PDF loading, chunking
        
        :param path: Path of PDF
        :param chunk_size: text chunking size
        :param chunk_overlap: chunks overlapping size
        """

        self.path = path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.path_img = img

    def pdfLoader(self) : 
        loader = PyPDFLoader(self.path)
        docs = loader.load()
        return  docs
    def txtLodaer(self) : 
        loader = TextLoader(self.path)
        docs = loader.load()
        return  docs
    def pdf_img_loader(self) : 
        img_base_path = "/Users/oussamayousr/Documents/ai-agent-multimodal-rag/data/colipali_data"
        img_path = []
        pdf_path = self.path
        images = convert_from_path(pdf_path)
        for i, image in enumerate(images):
            image.save(img_base_path+f"/page_{i + 1}.png", "PNG")
            img_path.append(img_base_path+f"/page_{i + 1}.png")
        return img_path , img_base_path 
        
    

        
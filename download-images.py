from pexel_downloader import PexelDownloader

if __name__ == '__main__':
    downloader = PexelDownloader(api_key="<YOUR-PEXELS-API-KEY>")

    query = "olympics sports"
    save_dir = "./dataset/images"
    downloader.download_images(query=query,
                               num_images=100,
                               save_directory=save_dir,
                               size='medium')
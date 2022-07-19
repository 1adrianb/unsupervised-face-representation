# flickr-scraper

Flickr Scraper allows you to search and download images from [Flickr](flickr.com/).

You need to configure the `.env` file with your API credentials (see `.env.example` for options).

## Setup

You require a Flickr API key that can be created here: [Flickr](https://www.flickr.com/services/apps/create/apply).

Copy the a `.env.example` file and rename to `.env` to create an environment file.

Add your `API_KEY` and `API_SECRET` to the `.env` file (see below for example).

```
API_KEY=xxxxx
API_SECRET=zzzzz
QUERY_TAGS=person,face
LICENSES=4,9
PER_PAGE=500
```

Run `npm install` to install all required dependencies.

## Running

You can run the script using `npm run start` to begin downloading the files.

## .env Configuration

The `.env` config file (see `.env.example` for example) contains a number of configurable options.

| Key        | Description                                                                                     | Example                          |
| ---------- | ----------------------------------------------------------------------------------------------- | -------------------------------- |
| API_KEY    | Your API Key (see setup)                                                                        | xxxxx |
| API_SECRET | Your API Secret (see setup)                                                                     | zzzzz                 |
| QUERY_TAGS | Comma separated list of tags to search accross (any tag match - more tags doesn't reduce query) | person,face                      |
| LICENSES   | Comma separated License IDs (see licenses below)                                                | 4,9                              |
| PER_PAGE   | Int (max 500) for your pagination size (default 500)                                            | 500                              |

## Data Output

The application creates `logs` and `downloads` (stored under the same logfile name and folder name in downloads) each time you run the script.

### Logs

The application will output log info in `/logs` so you can understand what is being queried.

The logfile has three parts: `params`, `pages`, `imageUrls`.

`params` tell you what query options were supplied to Flickr.

`pages` refers to the paginated results from the Flickr API response.

Photo object looks like this (to construct the download url see `File URLS` below):

```
{
  "id": "50437828752",
  "owner": "156445661@N02",
  "secret": "6d8b1d92a2",
  "server": "65535",
  "farm": 66,
  "title": "Woman in the gym doing elevated pushups while her trainer is watching.",
  "ispublic": 1,
  "isfriend": 0,
  "isfamily": 0
}
```

`imageUrls` are fully constructed urls to download the images returned from the initial request.

An example log file output can be seen below:

```
{
  "params": {
    "tags": "person,face",
    "media": "photos",
    "license": "4,9",
    "per_page": "50"
  },
  "pages": [
    {
      "photos": {
        "page": 1,
        "pages": 1218,
        "perpage": 50,
        "total": "60870",
        "photo": [
          {
            "id": "50437828752",
            "owner": "156445661@N02",
            "secret": "secret_id",
            "server": "65535",
            "farm": 66,
            "title": "Woman in the gym doing elevated pushups while her trainer is watching.",
            "ispublic": 1,
            "isfriend": 0,
            "isfamily": 0
          },
          {
            "id": "50436956808",
            "owner": "156445661@N02",
            "secret": "secret_id",
            "server": "65535",
            "farm": 66,
            "title": "Mechanic putting the oil stick back in it's place.",
            "ispublic": 1,
            "isfriend": 0,
            "isfamily": 0
          }
        ]
      }
    }
  ],
  "imageUrls": [
    "https://live.staticflickr.com/65535/{image_id}.jpg",
    "https://live.staticflickr.com/65535/{image_id}.jpg",
    "https://live.staticflickr.com/65535/{image_id}.jpg",
  ]
}

```

### Image Downloads

Images will be downloaded to `downloads/${timestamp}` where the timestamp format consists of `${YYYY-MM-DD hh:mm:ss}` (the same as the log filename).

All images are in `jpg` file type and are 500px on one dimention (default from Flickr).

Images use the format: `{server-id}_{id}_{secret}.jpg` which uses the data returned from the API request (see log data below).

![Example Image](example_image.jpg?raw=true)


## Flickr Docs

Useful info about working with Flickr data

## License types

You can select different license types as found here:
[https://www.flickr.com/services/api/explore/?method=flickr.photos.licenses.getInfo](https://www.flickr.com/services/api/explore/?method=flickr.photos.licenses.getInfo)

Commercial and public domain licenses are `4,9,10` and should be added to your `.env` file with the key `LICENSES` (see example env)

```
<?xml version="1.0" encoding="utf-8" ?>
<rsp stat="ok">
  <licenses>
    <license id="0" name="All Rights Reserved" url="" />
    <license id="4" name="Attribution License" url="https://creativecommons.org/licenses/by/2.0/" />
    <license id="6" name="Attribution-NoDerivs License" url="https://creativecommons.org/licenses/by-nd/2.0/" />
    <license id="3" name="Attribution-NonCommercial-NoDerivs License" url="https://creativecommons.org/licenses/by-nc-nd/2.0/" />
    <license id="2" name="Attribution-NonCommercial License" url="https://creativecommons.org/licenses/by-nc/2.0/" />
    <license id="1" name="Attribution-NonCommercial-ShareAlike License" url="https://creativecommons.org/licenses/by-nc-sa/2.0/" />
    <license id="5" name="Attribution-ShareAlike License" url="https://creativecommons.org/licenses/by-sa/2.0/" />
    <license id="7" name="No known copyright restrictions" url="https://www.flickr.com/commons/usage/" />
    <license id="8" name="United States Government Work" url="http://www.usa.gov/copyright.shtml" />
    <license id="9" name="Public Domain Dedication (CC0)" url="https://creativecommons.org/publicdomain/zero/1.0/" />
    <license id="10" name="Public Domain Mark" url="https://creativecommons.org/publicdomain/mark/1.0/" />
  </licenses>
</rsp>
```

### File URLS

You can access files by using the following structured URL:

```
https://live.staticflickr.com/{server-id}/{id}_{secret}.jpg
```

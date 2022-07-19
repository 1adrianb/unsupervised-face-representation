require('dotenv').config();
var fs = require("fs-extra");
var Flickr = require('flickr-sdk');
var flickr = new Flickr(process.env.API_KEY);
var axios = require('axios');

async function search(page = 1) {
  var results = [];
  try {
    results = await flickr.photos.search({
      tags: process.env.QUERY_TAGS, // tags to search across
      media: 'photos', // photos (can be 'videos' as well)
      license: process.env.LICENSES, // license type
      per_page: process.env.PER_PAGE, // default 100 / max 500 results per page
      page: page,
    });
  }
  catch (e) {
    console.log(e);
  }
  return results;
}

async function writeLog(data, filename) {
  await fs.ensureDir(`./logs`);
  fs.writeFileSync(
    `./logs/${filename}.json`,
    JSON.stringify(data, null, 2)
  );
}

function downloadFile(url, outputDest) {
  return axios.get(url, { responseType: 'stream' })
    .then((response) => {
      response.data.pipe(fs.createWriteStream(outputDest));
    })
    .catch((err) => {
      console.log(err, 'FILE DOWNLOAD ERROR');
    })
}

(async () => {
  try {
    // https://www.flickr.com/services/api/flickr.photos.search.html (docs)

    var result = await search();
    if (!result) {
      console.log('No results found');
      return;
    }

    var searchRequests = []
    console.log('Search Request Results');
    console.log(`Results: ${result.body.photos.total}`);
    console.log(`Pages: ${result.body.photos.pages}`);
    for (var i = 1; i <= result.body.photos.pages; i++) {
      // for (var i = 1; i <= 1; i++) {
      // add search requests to array of promises
      searchRequests.push(
        flickr.photos.search({
          tags: process.env.QUERY_TAGS, // tags to search across
          media: 'photos', // photos (can be 'videos' as well)
          license: process.env.LICENSES, // license type
          per_page: process.env.PER_PAGE, // default 100 / max 500 results per page
          page: i,
        }).then(r => r.body)
      );
    }
    console.log('Queries prepared:', result.body.photos.pages);
    Promise.all(searchRequests).then(async (pages) => {
      var timestamp = new Date()
        .toISOString()
        .replace(/T/, " ")
        .replace(/\..+/, "")
        .replace(/:/g, "-");

      var log = {};
      log.params = {
        tags: process.env.QUERY_TAGS, // tags to search across
        media: 'photos', // photos (can be 'videos' as well)
        license: process.env.LICENSES, // license type
        per_page: process.env.PER_PAGE, // default 100 / max 500 results per page
      };
      log.pages = pages;
      log.imageUrls = [];
      pages.forEach(p => {
        // some results are funky and are empty from request (might be rate limiting)
        if ('photos' in p && p.photos.photo.length > 0) {
          p.photos.photo.forEach(photo => {
            log.imageUrls.push(`https://live.staticflickr.com/${photo.server}/${photo.id}_${photo.secret}.jpg`);
          });
        }
      });

      var timestamp = new Date()
        .toISOString()
        .replace(/T/, " ")
        .replace(/\..+/, "")
        .replace(/:/g, "-");

      // log the pages
      console.log(`Writing debug log in /logs/${timestamp}.json`);
      await writeLog(log, timestamp);
      // make sure we have a dir
      await fs.ensureDir(`./downloads/${timestamp}`);
      // download images
      var downloadPromises = [];
      log.imageUrls.forEach((url) => {
        var filename = url.replace('https://live.staticflickr.com/', '');
        filename = filename.replace('/', '_');
        downloadPromises.push(downloadFile(url, `./downloads/${timestamp}/${filename}`));
      });

      console.log(`Downloading images to /downloads/${timestamp}/`);
      Promise.all(downloadPromises).then((downloads) => {
        console.log(`Downloaded ${log.imageUrls.length} images`);
      });

    });
  } catch (e) {
    console.log(e);
  }
})();

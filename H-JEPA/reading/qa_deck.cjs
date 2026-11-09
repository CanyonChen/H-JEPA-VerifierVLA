const fs = require('fs');
const path = require('path');
const {pathToFileURL} = require('url');
const {chromium} = require('C:/Users/SXF-Admin/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');

(async () => {
  const root = path.resolve(__dirname, '../repo');
  const browser = await chromium.launch({executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true});
  const page = await browser.newPage({viewport:{width:1920,height:1080},deviceScaleFactor:1});
  const errors = [];
  page.on('pageerror', e => errors.push(e.message));
  await page.goto(pathToFileURL(path.join(root,'deck.html')).href);
  await page.waitForFunction(() => [...document.images].every(i=>i.complete));
  await page.evaluate(() => document.fonts.ready);
  await page.mouse.move(1900, 1060);
  await page.waitForTimeout(300);
  const total = await page.locator('.slide').count();
  const pages = [];
  for (let n=1;n<=total;n++) {
    if (n>1) await page.keyboard.press('ArrowRight');
    await page.waitForTimeout(150);
    await page.screenshot({path:path.join(root,'preview',`slide-${String(n).padStart(2,'0')}.png`)});
    pages.push(await page.evaluate(() => {
      const sl=document.querySelector('.slide.active');
      const stage=document.querySelector('#stage').getBoundingClientRect();
      const bad=[];
      for (const e of sl.querySelectorAll('h1,.sub,.kicker,td,th,.li,.row,.n,.l,pre,svg text')) {
        const r=e.getBoundingClientRect();
        if (!r.width || !r.height) continue;
        if (r.left < stage.left-2 || r.top < stage.top-2 || r.right > stage.right+2 || r.bottom > stage.bottom+2)
          bad.push({tag:e.tagName,text:e.textContent.slice(0,80),x:r.x,y:r.y,w:r.width,h:r.height});
      }
      return {title:sl.dataset.title,section:sl.dataset.sec,overflow:bad,
        auxiliaryText:sl.querySelectorAll('.kicker,.sub,.footnote,.source,.caption').length,
        missingImages:[...sl.querySelectorAll('img')].filter(i=>!i.naturalWidth).map(i=>i.getAttribute('src'))};
    }));
  }
  await page.keyboard.press('Home');
  await page.waitForTimeout(50);
  const home=await page.evaluate(()=>location.hash);
  await page.keyboard.press('ArrowRight');
  await page.waitForTimeout(50);
  const next=await page.evaluate(()=>location.hash);
  await page.keyboard.press('End');
  await page.waitForTimeout(50);
  const end=await page.evaluate(()=>location.hash);
  await page.keyboard.press('g');
  const nav=await page.locator('#nav-panel').evaluate(el=>getComputedStyle(el).display!=='none');
  await page.keyboard.press('g');
  const links=await page.evaluate(()=>[...document.querySelectorAll('img,video,source,script[src],link[href]')].map(e=>e.getAttribute('src')||e.getAttribute('href')).filter(Boolean));
  const idleHudText=await page.locator('#hud').textContent();
  const report={total,pages,errors,navigation:{home,next,end,nav},idleHudText,remoteRuntimeResources:links.filter(x=>/^https?:/i.test(x)),
    passed:pages.every(p=>!p.overflow.length&&!p.missingImages.length&&!p.auxiliaryText)&&!errors.length&&!idleHudText&&home==='#1'&&next==='#2'&&end==='#'+total&&nav};
  fs.writeFileSync(path.join(root,'validation.json'),JSON.stringify(report,null,2));
  console.log(JSON.stringify({total,errors,navigation:report.navigation,overflow:pages.filter(p=>p.overflow.length),missing:pages.filter(p=>p.missingImages.length),remote:report.remoteRuntimeResources,passed:report.passed}));
  await browser.close();
  if(!report.passed)process.exitCode=1;
})().catch(e=>{console.error(e);process.exitCode=1;});

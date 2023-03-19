import type { CheerioAPI, load as LoadT } from 'cheerio';
import { Document } from 'langchain/document';
import { BaseDocumentLoader } from 'langchain/document_loaders';
import type { DocumentLoader } from 'langchain/document_loaders';
import { CheerioWebBaseLoader } from 'langchain/document_loaders';

export class CustomWebLoader
  extends BaseDocumentLoader
  implements DocumentLoader
{
  constructor(public webPath: string) {
    super();
  }

  static async _scrape(url: string): Promise<CheerioAPI> {
    const { load } = await CustomWebLoader.imports();
    const response = await fetch(url);
    const html = await response.text();
    return load(html);
  }

  async scrape(): Promise<CheerioAPI> {
    return CustomWebLoader._scrape(this.webPath);
  }

  async load(): Promise<Document[]> {
    const $ = await this.scrape();
    const title = [
      $('h1.ori-inshgit_header-title').text(),
      $('h1.ori-hero_section-title').text(),
    ].join('\n\n')

    const date = $('ul.ori-inshgit_meta-details li:first').text();

    const content = [
      $('p.ori-hero_section-description').text(),
      $('.ori-insight_content').text(),
      $('.text-image-content').text(),
      $('.items-post').text(),
      $('#ori-text_block').text(),
      $('#ori-testimonial_block .content').text(),
      $('.eleyton-header').text(),
      $('#ori-service_block').text(),
      $('.bleu-card-body').text(),
      $('#solo-post-list .saling-card-body').text(),
      $('.ori-text-image-header').text(),
      $('#ori-block_key').text(),
      $('#ori-icon_boxes-block').text(),
      $('#ori-why_choose_leyton_block').text(),
      $('#ori-cards_block').text(),
      $('#ori-location').text(),
      $('#ori-discover_industry_block').text(),
      $('.blog-container').text(),
    ].join('\n\n')

    const cleanedContent = content.replace(/\s+/g, ' ').trim();

    const contentLength = cleanedContent?.match(/\b\w+\b/g)?.length ?? 0;

    const metadata = { source: this.webPath, title, date, contentLength };

    return [new Document({ pageContent: cleanedContent, metadata })];
  }

  static async imports(): Promise<{
    load: typeof LoadT;
  }> {
    try {
      const { load } = await import('cheerio');
      return { load };
    } catch (e) {
      console.error(e);
      throw new Error(
        'Please install cheerio as a dependency with, e.g. `yarn add cheerio`',
      );
    }
  }
}
